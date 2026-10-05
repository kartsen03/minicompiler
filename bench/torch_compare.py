#!/usr/bin/env python3
"""minicompiler against PyTorch eager and torch.compile on the same graphs.

For each benchmark graph and size, several ways to compute the same function,
all in this one process, on the same seeded inputs:

  minicompiler        the graph compiled with all passes for the Eigen CPU
                      backend (--device cpu) or the CUDA backend (--device
                      cuda), called through bench/capi.cpp with ctypes
  eager               PyTorch running the graph op by op, exactly as written
                      (bench/mcgraph.py turns the .mcg file into Python)
  eager_idiomatic     the same function written the way a PyTorch user would,
                      with PyTorch's fused library ops: F.gelu(approximate="tanh"),
                      torch.addmm, F.batch_norm
  compile             torch.compile (Inductor, default mode) of the op-by-op
                      function
  compile_cudagraphs  GPU only: torch.compile with mode="reduce-overhead",
                      which replays the kernels as a CUDA graph to remove most
                      of the launch overhead

On the CPU, minicompiler runs on the NumPy buffers in place and each call is
timed with time.perf_counter_ns. On the GPU every variant's inputs live on the
device (minicompiler's are uploaded once), minicompiler enqueues its kernels on
the same CUDA stream as PyTorch, and each call is timed with CUDA events
recorded on that stream, from before the call to the end of its last kernel.
The host's launch overhead therefore counts whenever the GPU has to wait for
it, as it would for a real caller. Matmuls run in full FP32 everywhere (TF32 is
disabled in PyTorch and by NVIDIA_TF32_OVERRIDE=0).

Timing is interleaved: after warmup, each round runs every variant once in
rotating order. On a laptop a core's speed drifts (the same workload ran at
~1.1 ms, then 2-3 ms, then ~1.5 ms within one minute here), and a GPU's clock
moves with power and heat, so runs taken at different times cannot be
compared. Within a round all variants see the same conditions, so each round
gives a ratio PyTorch time / minicompiler time, and the median of those ratios
is the headline comparison. Every variant's output is checked against
minicompiler's.

    python bench/torch_compare.py --capi build/bench/libminicompiler_capi.so \\
        --out results/cpu/torch_compare.json
    python bench/torch_compare.py --device cuda --capi build/bench/libminicompiler_capi.so \\
        --out results/gpu/torch_compare.json
"""

from __future__ import annotations

import argparse
import contextlib
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

CPU_CONFIGS = [
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
# The first one is launch-dominated: 16 KB is a few microseconds of DRAM time.
GPU_CONFIGS = [
    ("gelu_chain", {"M": 1, "N": 4096}, "1x4096 (16 KB)"),
    ("gelu_chain", {"M": 256, "N": 4096}, "256x4096 (4 MB)"),
    ("gelu_chain", {"M": 2048, "N": 4096}, "2048x4096 (32 MB)"),
    ("gelu_chain", {"M": 8192, "N": 4096}, "8192x4096 (128 MB)"),
    ("matmul_bias_relu", {"M": 256, "K": 1024, "N": 1024}, "256x1024 @ 1024x1024"),
    ("matmul_bias_relu", {"M": 2048, "K": 1024, "N": 1024}, "2048x1024 @ 1024x1024"),
    ("mlp_block", {"B": 128, "D": 512, "H": 2048}, "B=128, 512->2048->512"),
    ("mlp_block", {"B": 512, "D": 512, "H": 2048}, "B=512, 512->2048->512"),
    ("mlp_block", {"B": 4096, "D": 512, "H": 2048}, "B=4096, 512->2048->512"),
]


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
    """A graph compiled through bench/capi.cpp for the CPU, run on NumPy
    buffers in place."""

    def __init__(self, lib, path: Path, dims: dict, inputs, output_shapes):
        import numpy as np

        self.lib = lib
        self.handle = compile_graph(lib, path, dims, b"cpu")
        self.inputs = [np.ascontiguousarray(a, dtype=np.float32) for a in inputs]  # kept alive
        self.outputs = [np.empty(shape, dtype=np.float32) for shape in output_shapes]
        check_output_sizes(lib, self.handle, self.outputs)
        self.in_ptrs = pointers(self.inputs)
        self.out_ptrs = pointers(self.outputs)

    def __call__(self):
        if self.lib.mc_run(self.handle, self.in_ptrs, self.out_ptrs) != 0:
            raise RuntimeError("mc_run failed")
        return self.outputs

    def results(self):
        return [o.copy() for o in self()]

    def close(self):
        self.lib.mc_release(self.handle)


class MinicompilerGpu:
    """A graph compiled through bench/capi.cpp for the CUDA backend. The inputs
    are uploaded once; each call enqueues the kernels on `stream` (PyTorch's)
    and leaves the outputs on the device."""

    def __init__(self, lib, path: Path, dims: dict, inputs, output_shapes, stream: int):
        import numpy as np

        if not stream:
            raise ValueError("needs a non-default CUDA stream")  # 0 would mean minicompiler's own stream
        self.lib = lib
        self.handle = compile_graph(lib, path, dims, b"cuda")
        self.output_shapes = output_shapes
        check_output_sizes(lib, self.handle, [np.empty(s, dtype=np.float32) for s in output_shapes])
        host_inputs = [np.ascontiguousarray(a, dtype=np.float32) for a in inputs]
        if lib.mc_upload(self.handle, pointers(host_inputs)) != 0:
            raise RuntimeError("mc_upload failed")
        self.stream = ctypes.c_void_p(stream)
        self.kernel_launches = lib.mc_kernel_launches(self.handle)
        names = ctypes.create_string_buffer(1024)
        lib.mc_matmul_kernels(self.handle, names, len(names))
        self.matmul_kernels = [n for n in names.value.decode().split(",") if n]

    def __call__(self):
        if self.lib.mc_enqueue(self.handle, self.stream) != 0:
            raise RuntimeError("mc_enqueue failed")

    def results(self):
        """Runs the graph once and copies its outputs to the host."""
        import numpy as np

        self()
        outputs = [np.empty(shape, dtype=np.float32) for shape in self.output_shapes]
        if self.lib.mc_download(self.handle, pointers(outputs)) != 0:
            raise RuntimeError("mc_download failed")
        return outputs

    def close(self):
        self.lib.mc_release(self.handle)


def compile_graph(lib, path: Path, dims: dict, backend: bytes):
    error = ctypes.create_string_buffer(1024)
    spec = ",".join(f"{k}={v}" for k, v in dims.items()).encode()
    handle = lib.mc_compile(str(path).encode(), spec, b"default", backend, error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode())
    return handle


def check_output_sizes(lib, handle, outputs) -> None:
    for k, out in enumerate(outputs):
        if lib.mc_output_elements(handle, k) != out.size:
            raise RuntimeError(f"output {k} size mismatch")


def pointers(arrays):
    return (ctypes.c_void_p * len(arrays))(*[a.ctypes.data for a in arrays])


def load_capi(path: str):
    lib = ctypes.CDLL(path)
    lib.mc_compile.restype = ctypes.c_void_p
    lib.mc_compile.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p,
                               ctypes.c_int]
    lib.mc_output_elements.restype = ctypes.c_longlong
    lib.mc_output_elements.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.mc_run.restype = ctypes.c_int
    lib.mc_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    lib.mc_upload.restype = ctypes.c_int
    lib.mc_upload.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.mc_enqueue.restype = ctypes.c_int
    lib.mc_enqueue.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.mc_download.restype = ctypes.c_int
    lib.mc_download.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.mc_kernel_launches.restype = ctypes.c_int
    lib.mc_kernel_launches.argtypes = [ctypes.c_void_p]
    lib.mc_matmul_kernels.restype = ctypes.c_int
    lib.mc_matmul_kernels.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.mc_gpu_sm_clock_mhz.restype = ctypes.c_double
    lib.mc_gpu_sm_clock_mhz.argtypes = []
    lib.mc_gpu_device_json.restype = ctypes.c_int
    lib.mc_gpu_device_json.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
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


def wall_clock_timer():
    def timed(fn) -> float:
        t0 = time.perf_counter_ns()
        fn()
        return (time.perf_counter_ns() - t0) / 1e6

    return timed


def cuda_event_timer():
    """Times one call with CUDA events on the current stream."""
    import torch

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    def timed(fn) -> float:
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        return start.elapsed_time(stop)

    return timed


def time_interleaved(fns: dict, warmup: int, warmup_seconds: float, seconds: float, timed,
                     after_round=None) -> dict[str, list[float]]:
    """Warms every function up (`warmup` calls each, then rounds of all of them
    until `warmup_seconds` have passed: a laptop GPU idles at a few hundred MHz
    and boosts only under sustained load), then runs rounds of one call each,
    rotating the order every round. Returns the per-round times in ms for each
    function."""
    names = list(fns)
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    until = time.perf_counter() + warmup_seconds
    while time.perf_counter() < until:
        for n in names:
            timed(fns[n])
    probe = sum(statistics.median(timed(fns[n]) for _ in range(3)) for n in names)
    rounds = max(20, min(2000, int(seconds * 1000.0 / max(probe, 1e-3))))
    samples = {n: [] for n in names}
    for r in range(rounds):
        for i in range(len(names)):
            n = names[(i + r) % len(names)]
            samples[n].append(timed(fns[n]))
        if after_round:
            after_round()
    return samples


def environment(threads: int, device: str) -> dict:
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
        "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "numpy": numpy.__version__,
        "torch_threads": threads,
        "host_power_plan": os.environ.get("MINICOMPILER_HOST_POWER_PLAN", "unknown"),
        "inductor_compile_threads": os.environ.get("TORCHINDUCTOR_COMPILE_THREADS"),
        # glibc returns large freed blocks to the OS by default, so each new
        # output tensor starts as fresh pages and pays page faults.
        "malloc_mmap_threshold": os.environ.get("MALLOC_MMAP_THRESHOLD_", "glibc default"),
    }
    if device == "cuda":
        import triton

        env["torch_cuda"] = torch.version.cuda
        env["triton"] = triton.__version__
        env["gpu"] = torch.cuda.get_device_name(0)
        env["tf32"] = {"torch.backends.cuda.matmul.allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "float32_matmul_precision": torch.get_float32_matmul_precision(),
                       "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE")}
    return env


def gpu_device(lib, clocks: list[float]) -> dict:
    out = ctypes.create_string_buffer(4096)
    samples = (ctypes.c_double * len(clocks))(*clocks)
    if lib.mc_gpu_device_json(samples, len(clocks), out, len(out)) != 0:
        return {}
    return json.loads(out.value.decode())["device"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capi", required=True, help="path to libminicompiler_capi.so")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="PyTorch intra-op threads (default 1, like Eigen here)")
    ap.add_argument("--seconds", type=float, default=8.0, help="timing budget per config, all variants together")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--warmup-seconds", type=float, default=None,
                    help="keep warming up for this long per config (default 1.0 on the GPU, 0 on the CPU)")
    ap.add_argument("--only", help="run only this graph")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    gpu = args.device == "cuda"
    if args.warmup_seconds is None:
        args.warmup_seconds = 1.0 if gpu else 0.0

    # Must be set before torch starts its thread pools and CUDA.
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    # Compile in this process instead of in a pool of worker processes that
    # would stay alive and compete for the timed core.
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    if gpu:
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"  # plain FP32 in every library, like minicompiler
    import torch

    import mcgraph

    torch.set_num_threads(args.threads)
    if gpu:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        stream = torch.cuda.Stream()
    lib = load_capi(args.capi)
    timed = cuda_event_timer() if gpu else wall_clock_timer()
    clocks: list[float] = []
    sample_clock = (lambda: clocks.append(lib.mc_gpu_sm_clock_mhz())) if gpu else None
    variants = ["eager", "eager_idiomatic", "compile"] + (["compile_cudagraphs"] if gpu else [])

    results = []
    widths = {v: max(len(v), 8) + 2 for v in variants}
    print(f"{'graph':17} {'size':24} {'minicompiler':>12}   PyTorch / minicompiler (median of per-round ratios)")
    print(f"{'':17} {'':24} {'median ms':>12}   " + "".join(f"{v:>{widths[v]}}" for v in variants))
    for name, dims, size in GPU_CONFIGS if gpu else CPU_CONFIGS:
        if args.only and name != args.only:
            continue
        path = HERE / "graphs" / f"{name}.mcg"
        g = mcgraph.parse(str(path), dims)
        inputs_np = mcgraph.random_inputs(g, seed=1)

        with torch.inference_mode(), torch.cuda.stream(stream) if gpu else contextlib.nullcontext():
            inputs = [torch.from_numpy(a).to(args.device) for a in inputs_np]
            eager_fn, scope = mcgraph.build_torch_fn(g, args.device)
            reference = [t.cpu().numpy().copy() for t in eager_fn(*inputs)]
            shapes = [r.shape for r in reference]
            mc = (MinicompilerGpu(lib, path, dims, inputs_np, shapes, stream.cuda_stream) if gpu else
                  Minicompiler(lib, path, dims, inputs_np, shapes))
            fns = {
                "minicompiler": mc,
                "eager": lambda f=eager_fn: f(*inputs),
                "eager_idiomatic": lambda f=idiomatic_fn(name, scope): f(*inputs),
            }
            compile_seconds = {}
            modes = {"compile": "default", "compile_cudagraphs": "reduce-overhead"}
            for v in variants[2:]:
                t0 = time.perf_counter()
                compiled = torch.compile(eager_fn, dynamic=False, fullgraph=True, mode=modes[v])
                compiled(*inputs)  # compiles
                if gpu:
                    torch.cuda.synchronize()
                compile_seconds[v] = time.perf_counter() - t0
                fns[v] = lambda f=compiled: f(*inputs)

            mc_out = mc.results()
            entry = {"graph": name, "size": size, "dims": dims}
            if gpu:
                # What the timer reports for a call that launches nothing: the
                # Python and event-recording overhead every variant includes.
                fns["timer_floor"] = lambda: None
            first_clock = len(clocks)
            samples = time_interleaved(fns, args.warmup, args.warmup_seconds, args.seconds, timed, sample_clock)
            entry["minicompiler"] = {"timing": summarize(samples["minicompiler"], args.warmup)}
            if gpu:
                entry["timer_floor_ms"] = summarize(samples["timer_floor"], args.warmup)["median_ms"]
                entry["minicompiler"]["kernel_launches"] = mc.kernel_launches
                entry["minicompiler"]["matmul_kernels"] = mc.matmul_kernels
                shape_clocks = sorted(clocks[first_clock:])
                entry["sm_clock_mhz_median"] = shape_clocks[len(shape_clocks) // 2] if shape_clocks else 0.0
            for v in variants:
                outs = [t.cpu().numpy() for t in fns[v]()]
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
            for v, seconds in compile_seconds.items():
                entry[v]["compile_seconds"] = seconds
            mc.close()
        results.append(entry)
        r = "".join(f"{entry[v]['ratio_to_minicompiler']['median']:{widths[v] - 1}.2f}x" for v in variants)
        print(f"{name:17} {size:24} {entry['minicompiler']['timing']['median_ms']:12.4f}   {r}")
        del fns, compiled, inputs, eager_fn, scope
        torch._dynamo.reset()
        if gpu:
            torch.cuda.empty_cache()

    if gpu:
        method = (f"all variants in one process on one CUDA stream; {args.warmup} warmup runs each and "
                  f"{args.warmup_seconds} s of warm-up rounds (the GPU boosts only under sustained load), then "
                  "interleaved rounds (one run of each variant per round, rotating order, each followed by a "
                  f"synchronize) for about {args.seconds} s per config; each run timed with CUDA events recorded "
                  "on the stream before and after the call, so "
                  "launch overhead counts when the GPU waits for the host; timing = median over rounds; "
                  "ratio_to_minicompiler = median over rounds of (variant time / minicompiler time) in the same "
                  "round. Inputs are on the device for every variant (minicompiler's uploaded once, then only "
                  "its kernels are enqueued, through ctypes); PyTorch runs under torch.inference_mode(); "
                  "FP32 matmuls without TF32 everywhere. timer_floor_ms is the same measurement of a call that "
                  "launches nothing, interleaved in the same rounds: the overhead included in every time.")
    else:
        method = (f"all variants in one process; {args.warmup} warmup runs each, then interleaved rounds (one run "
                  f"of each variant per round, rotating order) for about {args.seconds} s per config; "
                  "time.perf_counter_ns per run; timing = median over rounds; ratio_to_minicompiler = median over "
                  "rounds of (variant time / minicompiler time) in the same round. minicompiler is called through "
                  "ctypes on the NumPy buffers (no copies); PyTorch runs under torch.inference_mode().")
    doc = {
        "benchmark": f"minicompiler vs PyTorch eager and torch.compile ({'GPU' if gpu else 'CPU'})",
        "method": method,
        "environment": environment(args.threads, args.device),
    }
    if gpu:
        doc["device"] = gpu_device(lib, clocks)
    doc["results"] = results
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
