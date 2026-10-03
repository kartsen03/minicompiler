#!/usr/bin/env python3
"""minicompiler against PyTorch eager and torch.compile on the same graphs.

For each benchmark graph and size, four ways to compute the same function:

  minicompiler     `mcc --bench` with all passes; timed inside mcc
  eager            PyTorch running the graph op by op, exactly as written
                   (bench/mcgraph.py turns the .mcg file into Python)
  eager_idiomatic  the same function written the way a PyTorch user would,
                   with PyTorch's fused library ops: F.gelu(approximate="tanh"),
                   torch.addmm, F.batch_norm
  compile          torch.compile (Inductor, default mode) of the op-by-op function

Every variant is timed the same way (warmup runs, then individually timed
runs, median reported) on identical seeded inputs, and its outputs are
checked against minicompiler's. CPU runs are single-threaded on both sides
by default, because minicompiler's Eigen backend uses one thread.

    python bench/torch_compare.py --mcc build/tools/mcc --out results/cpu/torch_compare.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
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


def summarize(samples_ms: list[float], warmup: int) -> dict:
    s = sorted(samples_ms)

    def pct(q: float) -> float:
        pos = q * (len(s) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (pos - lo) * (s[hi] - s[lo])

    return {
        "warmup": warmup,
        "reps": len(s),
        "median_ms": pct(0.5),
        "min_ms": s[0],
        "p10_ms": pct(0.1),
        "p90_ms": pct(0.9),
        "mean_ms": statistics.fmean(s),
    }


def reps_for(single_ms: float, seconds: float) -> int:
    return max(20, min(2000, int(seconds * 1000.0 / max(single_ms, 1e-3))))


def time_torch(fn, args, device: str, warmup: int, seconds: float) -> dict:
    import torch

    def once() -> float:
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            end.synchronize()
            return start.elapsed_time(end)
        t0 = time.perf_counter_ns()
        fn(*args)
        return (time.perf_counter_ns() - t0) / 1e6

    for _ in range(warmup):
        fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    probe = statistics.median(once() for _ in range(5))
    samples = [once() for _ in range(reps_for(probe, seconds))]
    return summarize(samples, warmup)


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


def run_mcc(mcc: str, graph: Path, dims: dict, backend: str, extra: list[str]) -> dict:
    cmd = [mcc, str(graph), "--backend", backend] + [a for k, v in dims.items() for a in ("--dim", f"{k}={v}")]
    out = subprocess.run(cmd + extra, check=True, capture_output=True, text=True).stdout
    return json.loads(out) if "--bench" in extra else {}


def environment(device: str, threads: int) -> dict:
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
    env = {
        "commit": os.environ.get("MINICOMPILER_COMMIT", "unknown"),
        "cpu": cpu,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "torch_threads": threads,
        "host_power_plan": os.environ.get("MINICOMPILER_HOST_POWER_PLAN", "unknown"),
        "inductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        # glibc returns large freed blocks to the OS by default, so each new
        # output tensor starts as fresh pages and pays page faults.
        "malloc_mmap_threshold": os.environ.get("MALLOC_MMAP_THRESHOLD_", "glibc default"),
    }
    if device == "cuda":
        env["gpu"] = torch.cuda.get_device_name()
        env["torch_cuda"] = torch.version.cuda
        env["torch_matmul_tf32"] = torch.backends.cuda.matmul.allow_tf32
    return env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mcc", required=True, help="path to the mcc binary")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="PyTorch intra-op threads on CPU (default 1)")
    ap.add_argument("--seconds", type=float, default=2.0, help="timing budget per variant and config")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--only", help="run only this graph")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Must be set before torch starts its thread pools.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    # Compile in this process. Inductor otherwise starts a pool of compile
    # workers that stay alive and, under taskset, share the timed core: they
    # slowed every later measurement, minicompiler's included, by 1.5-2x.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    import numpy as np
    import torch

    import mcgraph

    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False  # compare FP32 with FP32
    torch.backends.cudnn.allow_tf32 = False
    backend = "cuda" if args.device == "cuda" else "cpu"

    results = []
    print(f"{'graph':17} {'size':24} {'minicompiler':>13} {'eager':>10} {'idiomatic':>10} {'compile':>10}  (median ms)")
    for name, dims, size in CONFIGS:
        if args.only and name != args.only:
            continue
        path = HERE / "graphs" / f"{name}.mcg"
        g = mcgraph.parse(str(path), dims)
        inputs_np = mcgraph.random_inputs(g, seed=1)
        inputs = [torch.from_numpy(a).to(args.device) for a in inputs_np]

        # minicompiler: dump outputs for the cross-check, then benchmark.
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "out.bin"
            probe = run_mcc(args.mcc, path, dims, backend, ["--seed", "1", "--dump-outputs", str(dump),
                                                            "--bench", "--warmup", "3", "--reps", "5"])
            flat = np.fromfile(dump, dtype=np.float32)
        reps = reps_for(probe["timing"]["median_ms"], args.seconds)
        mc = run_mcc(args.mcc, path, dims, backend, ["--seed", "1", "--bench", "--warmup", str(args.warmup),
                                                     "--reps", str(reps)])
        mc_outputs, offset = [], 0
        with torch.inference_mode():
            eager_fn, scope = mcgraph.build_torch_fn(g, args.device)
            expected_shapes = [t.shape for t in eager_fn(*inputs)]
        for shape in expected_shapes:
            n = int(np.prod(shape))
            mc_outputs.append(flat[offset:offset + n].reshape(shape))
            offset += n

        entry = {"graph": name, "size": size, "dims": dims, "minicompiler": {
            "timing": mc["timing"], "optimized_stats": mc["optimized_stats"], "input_stats": mc["input_stats"]}}
        with torch.inference_mode():
            variants = {
                "eager": eager_fn,
                "eager_idiomatic": idiomatic_fn(name, scope),
            }
            t0 = time.perf_counter()
            compiled = torch.compile(eager_fn, dynamic=False, fullgraph=True)
            compiled(*inputs)  # compiles
            compile_seconds = time.perf_counter() - t0
            variants["compile"] = compiled
            for vname, fn in variants.items():
                outs = [t.float().cpu().numpy() for t in fn(*inputs)]
                diff = normwise_diff(outs, mc_outputs)
                entry[vname] = {"timing": time_torch(fn, inputs, args.device, args.warmup, args.seconds),
                                "max_normwise_diff_vs_minicompiler": diff}
                if diff > 1e-4:
                    print(f"  WARNING: {vname} differs from minicompiler by {diff:.2e}", file=sys.stderr)
            entry["compile"]["compile_seconds"] = compile_seconds
        results.append(entry)
        med = {k: entry[k]["timing"]["median_ms"] for k in ("minicompiler", "eager", "eager_idiomatic", "compile")}
        print(f"{name:17} {size:24} {med['minicompiler']:13.4f} {med['eager']:10.4f} "
              f"{med['eager_idiomatic']:10.4f} {med['compile']:10.4f}")
        torch._dynamo.reset()

    doc = {
        "benchmark": f"minicompiler vs PyTorch eager and torch.compile ({args.device})",
        "method": f"{args.warmup} warmup runs, then individually timed runs for about {args.seconds} s per "
                  "variant; median reported. minicompiler is timed inside mcc (C++ steady_clock"
                  + (", CUDA events" if args.device == "cuda" else "")
                  + "), PyTorch in Python (" + ("CUDA events" if args.device == "cuda" else "time.perf_counter_ns")
                  + ") under torch.inference_mode().",
        "environment": environment(args.device, args.threads),
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
