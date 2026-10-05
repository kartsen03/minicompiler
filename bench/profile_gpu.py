#!/usr/bin/env python3
"""Runs the GPU comparison's variants under NVTX ranges for Nsight Systems.

Not a benchmark (bench/torch_compare.py is): it gives the profiler a labeled
sequence of calls. For each config, every variant is built and warmed up as in
torch_compare.py, then called `--calls` times, each call synchronized and
wrapped in an NVTX range named "<config>/<variant>". In the profile, a range's
duration is the call's wall time and the kernels inside it are the GPU's
work, so the difference is the host's share: launch overhead, Python and the
synchronization. scripts/profile_gpu.sh runs this under nsys and summarizes.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIGS = {
    "gelu_16kb": ("gelu_chain", {"M": 1, "N": 4096}),
    "gelu_32mb": ("gelu_chain", {"M": 2048, "N": 4096}),
    "mlp_b128": ("mlp_block", {"B": 128, "D": 512, "H": 2048}),
    "mlp_b512": ("mlp_block", {"B": 512, "D": 512, "H": 2048}),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capi", required=True, help="path to libminicompiler_capi.so")
    ap.add_argument("--calls", type=int, default=50, help="profiled calls per variant")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated, from: " + ", ".join(CONFIGS))
    args = ap.parse_args()

    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import torch

    import mcgraph
    import torch_compare as tc

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    lib = tc.load_capi(args.capi)
    stream = torch.cuda.Stream()
    for key in args.configs.split(","):
        name, dims = CONFIGS[key]
        path = HERE / "graphs" / f"{name}.mcg"
        g = mcgraph.parse(str(path), dims)
        inputs_np = mcgraph.random_inputs(g, seed=1)
        with torch.inference_mode(), torch.cuda.stream(stream):
            inputs = [torch.from_numpy(a).cuda() for a in inputs_np]
            eager_fn, scope = mcgraph.build_torch_fn(g, "cuda")
            shapes = [t.shape for t in eager_fn(*inputs)]
            mc = tc.MinicompilerGpu(lib, path, dims, inputs_np, shapes, stream.cuda_stream)
            fns = {"minicompiler": mc, "eager": lambda f=eager_fn: f(*inputs),
                   "eager_idiomatic": lambda f=tc.idiomatic_fn(name, scope): f(*inputs)}
            for v, mode in [("compile", "default"), ("compile_cudagraphs", "reduce-overhead")]:
                compiled = torch.compile(eager_fn, dynamic=False, fullgraph=True, mode=mode)
                fns[v] = lambda f=compiled: f(*inputs)
            # Warm up as the benchmark does: compile, then a second of rounds.
            for fn in fns.values():
                for _ in range(10):
                    fn()
            until = time.perf_counter() + 1.0
            while time.perf_counter() < until:
                for fn in fns.values():
                    fn()
                    torch.cuda.synchronize()
            for v, fn in fns.items():
                for _ in range(args.calls):
                    torch.cuda.nvtx.range_push(f"{key}/{v}")
                    fn()
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()
            mc.close()
        print(f"profiled {key}: {', '.join(fns)}")
        del fns, inputs, eager_fn, scope, compiled
        torch._dynamo.reset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
