#!/usr/bin/env python3
"""minicompiler against PyTorch eager and torch.compile on the same graphs.

For each benchmark graph and size, four ways to compute the same function, all
in this one process, on the same seeded inputs:

  minicompiler     the graph compiled with all passes for the Eigen CPU
                   backend, run on the NumPy arrays through bench/capi.cpp
                   (ctypes, no copies)
  eager            PyTorch running the graph op by op, exactly as written
                   (bench/mcgraph.py turns the .mcg file into Python)
  eager_idiomatic  the same function written the way a PyTorch user would,
                   with PyTorch's fused library ops: F.gelu(approximate="tanh"),
                   torch.addmm, F.batch_norm
  compile          torch.compile (Inductor, default mode) of the op-by-op function

Timing is interleaved: after warmup, each round runs every variant once in
rotating order. On a laptop a core's speed drifts (the same workload ran at
~1.1 ms, then 2-3 ms, then ~1.5 ms within one minute here), so runs taken at
different times cannot be compared. Within a round all variants see the same
conditions, so each round gives a ratio PyTorch time / minicompiler time, and
the median of those ratios is the headline comparison. Every variant's output
is checked against minicompiler's.

    python bench/torch_compare.py --capi build/bench/libminicompiler_capi.so \\
        --out results/cpu/torch_compare.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CONFIGS = [
    ("gelu_chain", {"M": 1, "N": 4096}, "1x4096 (16 KB)"),
    ("gelu_chain", {"M": 64, "N": 4096}, "64x4096 (1 MB)"),
    ("gelu_chain", {"M": 256, "N": 4096}, "256x4096 (4 MB)"),
    ("gelu_chain", {"M": 2048, "N": 4096}, "2048x4096 (32 MB)"),
    ("matmul_bias_relu", {"M": 64, "K": 1024, "N": 1024}, "64x1024 @ 1024x1024"),
    ("matmul_bias_relu", {"M": 256, "K": 1024, "N": 1024}, "256x1024 @ 1024x1024"),
    ("mlp_block", {"B": 32, "D": 512, "H": 2048}, "B=32, 512->2048->512"),
    ("mlp_block", {"B": 128, "D": 512, "H": 2048}, "B=128, 512->2048->512"),
    ("mlp_block", {"B": 512, "D": 512, "H": 2048}, "B=512, 512->2048->512"),
]
VARIANTS = ["minicompiler", "eager", "eager_idiomatic", "compile"]


def percentile(sorted_values: list[float], q: float) -> float:
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (pos - lo) * (sorted_values[hi] - sorted_values[lo])


def summarize(samples_ms: list[float], warmup: int) -> dict:
    s = sorted(samples_ms)
    return {"warmup": warmup, "reps": len(s), "median_ms": percentile(s, 0.5), "min_ms": s[0],
            "p10_ms": percentile(s, 0.1), "p90_ms": percentile(s, 0.9), "mean_ms": statistics.fmean(s)}


class Minicompiler:
    """A graph compiled through bench/capi.cpp, run on NumPy buffers in place."""

    def __init__(self, lib, path: Path, dims: dict, inputs, output_shapes):
        import numpy as np

        error = ctypes.create_string_buffer(1024)
        spec = ",".join(f"{k}={v}" for k, v in dims.items()).encode()
        self.lib = lib
        self.handle = lib.mc_compile(str(path).encode(), spec, b"default", error, len(error))
        if not self.handle:
            raise RuntimeError(error.value.decode())
        self.inputs = [np.ascontiguousarray(a, dtype=np.float32) for a in inputs]  # kept alive
        self.outputs = [np.empty(shape, dtype=np.float32) for shape in output_shapes]
        for k, out in enumerate(self.outputs):
            if lib.mc_output_elements(self.handle, k) != out.size:
                raise RuntimeError(f"output {k} size mismatch")
        self.in_ptrs = (ctypes.c_void_p * len(self.inputs))(*[a.ctypes.data for a in self.inputs])
        self.out_ptrs = (ctypes.c_void_p * len(self.outputs))(*[o.ctypes.data for o in self.outputs])

    def __call__(self):
        if self.lib.mc_run(self.handle, self.in_ptrs, self.out_ptrs) != 0:
            raise RuntimeError("mc_run failed")
        return self.outputs

    def close(self):
        self.lib.mc_release(self.handle)


def load_capi(path: str):
    lib = ctypes.CDLL(path)
    lib.mc_compile.restype = ctypes.c_void_p
    lib.mc_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.mc_output_elements.restype = ctypes.c_longlong
    lib.mc_output_elements.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mc_run.restype = ctypes.c_int
    lib.mc_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.mc_release.argtypes = [ctypes.c_void_p]
    return lib


def idiomatic_fn(name: str, scope: dict):
    """The graph written with PyTorch's fused library ops."""
    import torch
    import torch.nn.functional as F

    if name == "gelu_chain":
        return lambda x: (F.gelu(x, approximate="tanh"),)
    if name == "matmul_bias_relu":
        w, b = scope["w"], scope["b"]
        return lambda x: (torch.relu(torch.addmm(b, x, w)),)
    if name == "mlp_block":
        c = scope
        eps = float(c["eps"].item())

        def mlp(x):
            h = torch.addmm(c["b1"], x, c["w1"])
            bn = F.batch_norm(h, c["mean"], c["var"], c["gamma"], c["beta"], training=False, eps=eps)
            a = F.gelu(bn, approximate="tanh")
            return (torch.addmm(c["b2"], a, c["w2"]) + x,)

        return mlp
    raise ValueError(name)


def normwise_diff(actual, expected) -> float:
    import numpy as np

    worst = 0.0
    for a, e in zip(actual, expected):
        a = np.asarray(a, dtype=np.float64).ravel()
        e = np.asarray(e, dtype=np.float64).ravel()
        worst = max(worst, float(np.max(np.abs(a - e)) / max(1.0, float(np.max(np.abs(e))))))
    return worst


def time_interleaved(fns: dict, warmup: int, seconds: float) -> dict[str, list[float]]:
    """Warms every function up, then runs rounds of one call each, rotating the
    order every round. Returns the per-round times in ms for each function."""
    names = list(fns)
    for fn in fns.values():
        for _ in range(warmup):
            fn()

    def timed(fn) -> float:
        t0 = time.perf_counter_ns()
        fn()
        return (time.perf_counter_ns() - t0) / 1e6

    probe = sum(statistics.median(timed(fns[n]) for _ in range(3)) for n in names)
    rounds = max(20, min(2000, int(seconds * 1000.0 / max(probe, 1e-3))))
    samples = {n: [] for n in names}
    for r in range(rounds):
        for i in range(len(names)):
            n = names[(i + r) % len(names)]
            samples[n].append(timed(fns[n]))
    return samples


def environment(threads: int) -> dict:
    import numpy
    import torch

    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo", encoding="utf-8"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {
        "commit": os.environ.get("MINICOMPILER_COMMIT", "unknown"),
        "cpu": cpu,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "numpy": numpy.__version__,
        "torch_threads": threads,
        "host_power_plan": os.environ.get("MINICOMPILER_HOST_POWER_PLAN", "unknown"),
        "inductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        # glibc returns large freed blocks to the OS by default, so each new
        # output tensor starts as fresh pages and pays page faults.
        "malloc_mmap_threshold": os.environ.get("MALLOC_MMAP_THRESHOLD_", "glibc default"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capi", required=True, help="path to libminicompiler_capi.so")
    ap.add_argument("--threads", type=int, default=1, help="PyTorch intra-op threads (default 1, like Eigen here)")
    ap.add_argument("--seconds", type=float, default=8.0, help="timing budget per config, all variants together")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--only", help="run only this graph")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Must be set before torch starts its thread pools.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    # Compile in this process instead of in a pool of worker processes that
    # would stay alive and compete for the timed core.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import torch

    import mcgraph

    torch.set_num_threads(args.threads)
    lib = load_capi(args.capi)

    results = []
    print(f"{'graph':17} {'size':24} {'minicompiler':>12}   PyTorch / minicompiler (median of per-round ratios)")
    print(f"{'':17} {'':24} {'median ms':>12}   {'eager':>8} {'idiomatic':>10} {'compile':>8}")
    for name, dims, size in CONFIGS:
        if args.only and name != args.only:
            continue
        path = HERE / "graphs" / f"{name}.mcg"
        g = mcgraph.parse(str(path), dims)
        inputs_np = mcgraph.random_inputs(g, seed=1)
        inputs = [torch.from_numpy(a) for a in inputs_np]

        with torch.inference_mode():
            eager_fn, scope = mcgraph.build_torch_fn(g, "cpu")
            reference = [t.numpy().copy() for t in eager_fn(*inputs)]
            mc = Minicompiler(lib, path, dims, inputs_np, [r.shape for r in reference])
            fns = {
                "minicompiler": mc,
                "eager": lambda f=eager_fn: f(*inputs),
                "eager_idiomatic": lambda f=idiomatic_fn(name, scope): f(*inputs),
            }
            t0 = time.perf_counter()
            compiled = torch.compile(eager_fn, dynamic=False, fullgraph=True)
            compiled(*inputs)  # compiles
            compile_seconds = time.perf_counter() - t0
            fns["compile"] = lambda f=compiled: f(*inputs)

            mc_out = [o.copy() for o in mc()]
            entry = {"graph": name, "size": size, "dims": dims}
            samples = time_interleaved(fns, args.warmup, args.seconds)
            entry["minicompiler"] = {"timing": summarize(samples["minicompiler"], args.warmup)}
            for v in VARIANTS[1:]:
                outs = [t.numpy() for t in fns[v]()]
                ratios = sorted(a / b for a, b in zip(samples[v], samples["minicompiler"]))
                entry[v] = {
                    "timing": summarize(samples[v], args.warmup),
                    "ratio_to_minicompiler": {"median": percentile(ratios, 0.5), "p10": percentile(ratios, 0.1),
                                              "p90": percentile(ratios, 0.9)},
                    "max_normwise_diff_vs_minicompiler": normwise_diff(outs, mc_out),
                }
                if entry[v]["max_normwise_diff_vs_minicompiler"] > 1e-4:
                    print(f"  WARNING: {v} differs from minicompiler by "
                          f"{entry[v]['max_normwise_diff_vs_minicompiler']:.2e}", file=sys.stderr)
            entry["compile"]["compile_seconds"] = compile_seconds
            mc.close()
        results.append(entry)
        r = {v: entry[v]["ratio_to_minicompiler"]["median"] for v in VARIANTS[1:]}
        print(f"{name:17} {size:24} {entry['minicompiler']['timing']['median_ms']:12.4f}   "
              f"{r['eager']:7.2f}x {r['eager_idiomatic']:9.2f}x {r['compile']:7.2f}x")
        torch._dynamo.reset()

    doc = {
        "benchmark": "minicompiler vs PyTorch eager and torch.compile (CPU)",
        "method": f"all variants in one process; {args.warmup} warmup runs each, then interleaved rounds (one run "
                  f"of each variant per round, rotating order) for about {args.seconds} s per config; "
                  "time.perf_counter_ns per run; timing = median over rounds; ratio_to_minicompiler = median over "
                  "rounds of (variant time / minicompiler time) in the same round. minicompiler is called through "
                  "ctypes on the NumPy buffers (no copies); PyTorch runs under torch.inference_mode().",
        "environment": environment(args.threads),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
