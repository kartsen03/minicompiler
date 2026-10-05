#!/usr/bin/env python3
"""Speed and accuracy of tanh in minicompiler (Eigen) and PyTorch on the CPU.

PyTorch's F.gelu is already a single fused kernel, so on the GELU benchmark
much of the difference between minicompiler and PyTorch comes from how each
evaluates tanh, not from fusion. This measures that directly: worst-case
error against a float64 reference over 4M evenly spaced points in [-9, 9]
(where tanh is not yet saturated), and time per call with preallocated
outputs, interleaved in one process.

    python bench/tanh_check.py --capi build/bench/libminicompiler_capi.so --out results/cpu/tanh.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capi", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import numpy as np
    import torch

    from torch_compare import Minicompiler, environment, load_capi

    torch.set_num_threads(1)
    n = 1 << 22
    x = np.linspace(-9.0, 9.0, n, dtype=np.float32)
    ref = np.tanh(x.astype(np.float64))
    ulp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)

    def max_ulp(y) -> float:
        return float(np.max(np.abs(np.asarray(y, dtype=np.float64) - ref) / ulp))

    with tempfile.TemporaryDirectory() as tmp:
        graph = Path(tmp) / "tanh.mcg"
        graph.write_text(f"input x : f32[{n}]\ny = tanh x\noutput y\n", encoding="utf-8")
        mc = Minicompiler(load_capi(args.capi), graph, {}, [x], [(n,)])
        xt = torch.from_numpy(x)
        out = torch.empty_like(xt)
        fns = {"minicompiler_eigen": mc, "pytorch": lambda: torch.tanh(xt, out=out)}
        errors = {"minicompiler_eigen": max_ulp(mc()[0]), "pytorch": max_ulp(torch.tanh(xt).numpy())}
        for f in fns.values():
            for _ in range(5):
                f()
        samples = {k: [] for k in fns}
        for r in range(60):
            for k in (list(fns) if r % 2 else list(fns)[::-1]):
                t0 = time.perf_counter_ns()
                fns[k]()
                samples[k].append((time.perf_counter_ns() - t0) / 1e6)
        mc.close()

    ratios = sorted(a / b for a, b in zip(samples["pytorch"], samples["minicompiler_eigen"]))
    doc = {
        "benchmark": f"tanh over {n} float32 values in [-9, 9], one thread, preallocated outputs",
        "method": "max error in ulps of the float32 result against numpy float64 tanh; time: 5 warmup calls, "
                  "then 60 interleaved rounds, median; ratio = median over rounds of pytorch / minicompiler",
        "environment": environment(1),
        "max_error_ulp": errors,
        "median_ms": {k: statistics.median(v) for k, v in samples.items()},
        "pytorch_time_over_minicompiler": statistics.median(ratios),
    }
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: doc[k] for k in ("max_error_ulp", "median_ms", "pytorch_time_over_minicompiler")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
