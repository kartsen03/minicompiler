#!/usr/bin/env python3
"""Renders the result files under results/ as Markdown tables and writes them
into the README between markers, so no number in the README is typed by hand.

    python3 scripts/results_tables.py            # rewrite README.md in place
    python3 scripts/results_tables.py --print    # print the tables instead

A table is written between `<!-- BEGIN name -->` and `<!-- END name -->`.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(rel: str) -> dict | None:
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ms(v: float) -> str:
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.4f}"


def ratio(s: dict) -> str:
    """'2.92x (2.61-3.10)': median across runs, then the range."""
    return f"{s['median']:.2f}x ({s['min']:.2f}–{s['max']:.2f})"


def env_line(env: dict, keys: list[tuple[str, str]]) -> str:
    parts = [f"{label} {env[k]}" for k, label in keys if k in env]
    return "Measured on " + ", ".join(parts) + "."


def cpu_passes() -> str:
    d = load("results/cpu/passes.json")
    if not d:
        return "_No results recorded yet._"
    rows = ["| Graph | Size | Nodes | Compute nodes | Passes off | Passes on | Speedup (range over runs) |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for r in d["results"]:
        v = {x["variant"]: x for x in r["variants"]}
        none, full = v["none"], v["all"]
        rows.append(f"| `{r['graph']}` | {r['size']} | {none['nodes']} → {full['nodes']} | "
                    f"{none['compute_nodes']} → {full['compute_nodes']} | {ms(none['median_ms']['median'])} ms | "
                    f"{ms(full['median_ms']['median'])} ms | {ratio(full['speedup_paired'])} |")
    e = d["environment"]
    rows.append("")
    rows.append(f"Each of {d['runs']} runs times the variants in interleaved rounds; a run's speedup is the median "
                "over rounds of (passes-off time / passes-on time) within a round. The table shows the median "
                "across runs, the range across runs, and median times. "
                + env_line(e, [("cpu", "CPU:"), ("os", "OS:"), ("compiler", "compiler:"), ("eigen", "Eigen"),
                               ("build_flags", "build:"), ("threads", "threads:"),
                               ("host_power_plan", "Windows power plan:"), ("commit", "commit")])
                + " Raw runs and the per-pass ablation: `results/cpu/`.")
    return "\n".join(rows)


def cpu_torch(path: str = "results/cpu/torch_compare.json") -> str:
    d = load(path)
    if not d:
        return "_Not recorded yet._"
    rows = ["| Graph | Size | minicompiler | PyTorch eager (same ops) | PyTorch eager (idiomatic) | torch.compile |",
            "|---|---|---:|---:|---:|---:|"]
    for r in d["results"]:
        def cell(key: str) -> str:
            t = ms(r[key]["median_ms"]["median"])
            if key == "minicompiler":
                return f"{t} ms"
            return f"{t} ms, {ratio(r[key]['ratio_to_minicompiler'])}"
        rows.append(f"| `{r['graph']}` | {r['size']} | {cell('minicompiler')} | {cell('eager')} | "
                    f"{cell('eager_idiomatic')} | {cell('compile')} |")
    e = d["environment"]
    rows.append("")
    rows.append(f"Ratios are PyTorch time / minicompiler time (above 1 means minicompiler is faster): within each of "
                f"{d['runs']} runs, the median over interleaved rounds; shown as the median across runs with the "
                "range across runs. "
                + env_line(e, [("cpu", "CPU:"), ("torch", "PyTorch"), ("python", "Python"),
                               ("torch_threads", "threads:"), ("malloc_mmap_threshold", "glibc mmap threshold:"),
                               ("host_power_plan", "Windows power plan:"), ("commit", "commit")]))
    return "\n".join(rows)


def gpu_line(d: dict) -> str:
    """The GPU, its clocks during the runs, the software and the commit."""
    dev, env = d.get("device", {}), d["environment"]
    clocks = dev.get("sm_clock_mhz_during_run", {})
    parts = [f"GPU: {dev.get('name', env.get('gpu', 'unknown'))} ({dev.get('sm_count', '?')} SMs, compute capability "
             f"{dev.get('compute_capability', '?')})"]
    if clocks:
        median = clocks.get("median_of_run_medians", clocks.get("median"))
        parts.append(f"SM clock during the runs {clocks['min']:.0f}–{clocks['max']:.0f} MHz (median {median:.0f})")
    if "cuda_runtime" in dev:
        parts.append(f"CUDA runtime {dev['cuda_runtime'] // 1000}.{dev['cuda_runtime'] % 1000 // 10}, driver API "
                     f"{dev['cuda_driver'] // 1000}.{dev['cuda_driver'] % 1000 // 10}")
    for k, label in [("torch", "PyTorch"), ("torch_cuda", "PyTorch's CUDA"), ("triton", "Triton"),
                     ("compiler", "host compiler:"), ("host_power_plan", "Windows power plan:")]:
        if k in env:
            parts.append(f"{label} {env[k]}")
    commits = d.get("commits", [env.get("commit", "unknown")])
    parts.append("commit " + ", ".join(commits))
    return "Measured on " + "; ".join(parts) + "."


def gbs(s: dict, pct: dict) -> str:
    return f"{s['median']:.0f} GB/s ({pct['median']:.0f}%)"


def gpu_elementwise(results: str = "results/gpu") -> str:
    d = load(f"{results}/elementwise.json")
    if not d:
        return "_Not recorded yet._"
    rows = ["| GELU chain size | Unfused: kernels, time, bandwidth (% of peak) | Fused: time, bandwidth (% of peak) | "
            "Fused speedup (range over runs) |",
            "|---|---:|---:|---:|"]
    for r in d["results"]:
        v = {x["variant"]: x for x in r["variants"]}
        u, f = v["unfused"], v["fused"]
        rows.append(f"| {r['size']} | {u['kernel_launches']} kernels, {ms(u['median_ms']['median'])} ms, "
                    f"{gbs(u['achieved_bandwidth_gbs'], u['percent_of_peak_bandwidth'])} | "
                    f"{ms(f['median_ms']['median'])} ms, "
                    f"{gbs(f['achieved_bandwidth_gbs'], f['percent_of_peak_bandwidth'])} | "
                    f"{ratio(f['speedup_paired'])} |")
    peak = d.get("device", {}).get("peak_bandwidth_gbs")
    rows.append("")
    rows.append(f"Times are CUDA-event medians with the input already on the device, over {d['runs']} runs (median "
                "across runs). Bandwidth counts the bytes each kernel must read and write once"
                + (f"; peak = 2 × memory clock × bus width = {peak:.0f} GB/s, from the device properties" if peak else "")
                + ". " + gpu_line(d))
    return "\n".join(rows)


def gpu_matmul(results: str = "results/gpu") -> str:
    d = load(f"{results}/matmul.json")
    if not d:
        return "_Not recorded yet._"
    rows = ["| Shape (m × k × n) | Naive | Tiled | Register 128×128 | Register 64×64 | Vectorized | Double-buffered | "
            "Split-K | cuBLAS | Default: % of cuBLAS | vs naive | % of FP32 peak |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in d["results"]:
        k = {x["kernel"]: x for x in r["kernels"]}
        picked = r["auto_picks"]

        def g(name: str) -> str:
            text = f"{k[name]['gflops']['median']:,.0f}"
            return f"**{text}**" if name == picked else text

        splits = r["split_k_splits"]
        split_cell = f"{g('split_k')} ({splits} splits)" if splits > 1 else "(no split)"
        p = k[picked]
        rows.append(f"| {r['shape']} | {g('naive')} | {g('tiled')} | {g('register_tiled_128')} | "
                    f"{g('register_tiled_64')} | {g('vectorized')} | {g('double_buffered')} | {split_cell} | "
                    f"{g('cublas')} | {p['percent_of_cublas']['median']:.0f}% ({p['percent_of_cublas']['min']:.0f}–"
                    f"{p['percent_of_cublas']['max']:.0f}) | {p['speedup_over_naive']['median']:.2f}x | "
                    f"{p['percent_of_peak_at_measured_clock']['median']:.0f}% |")
    peak = d["peak_fp32_gflops"]
    near = sum(all(x >= 0.97 for x in r["auto_vs_fastest_per_run"]) for r in d["results"])
    rows.append("")
    rows.append(f"GFLOP/s = 2mnk / GPU time, median across {d['runs']} runs. Each variant's launches are timed on "
                "the GPU alone, between event nodes captured with them into a CUDA graph, so the host's launch "
                "overhead, which differs between cuBLAS and these kernels, is left out for all. Bold is the "
                "backend's default: split-K where the output is too small to fill the GPU, otherwise "
                f"double-buffered. In every run it was within 3% of the fastest of these kernels for {near} of "
                f"{len(d['results'])} shapes. Every kernel, cuBLAS included, is checked against a float64 reference "
                "within the FP32 error bound before timing; cuBLAS runs in plain FP32 without TF32. Peak FP32 = 2 × "
                f"SMs × FP32 lanes per SM × clock: {peak['at_max_sm_clock']:,.0f} GFLOP/s at the "
                f"{peak['max_sm_clock_mhz']:.0f} MHz maximum; the last column uses the median SM clock measured "
                "during each shape's runs. " + gpu_line(d))
    return "\n".join(rows)


def gpu_torch(results: str = "results/gpu") -> str:
    d = load(f"{results}/torch_compare.json")
    if not d:
        return "_Not recorded yet._"
    rows = ["| Graph | Size | minicompiler | PyTorch eager (same ops) | PyTorch eager (idiomatic) | torch.compile | "
            "torch.compile, CUDA graphs | SM clock |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    floors = []
    for r in d["results"]:
        def cell(key: str) -> str:
            t = ms(r[key]["median_ms"]["median"])
            if key == "minicompiler":
                return f"{t} ms"
            return f"{t} ms, {ratio(r[key]['ratio_to_minicompiler'])}"
        floors.append(r["timer_floor_ms"]["median"] * 1000)
        rows.append(f"| `{r['graph']}` | {r['size']} | {cell('minicompiler')} | {cell('eager')} | "
                    f"{cell('eager_idiomatic')} | {cell('compile')} | {cell('compile_cudagraphs')} | "
                    f"{r['sm_clock_mhz_median']['median']:.0f} MHz |")
    rows.append("")
    rows.append("Ratios are PyTorch time / minicompiler time (above 1 means minicompiler is faster): within each of "
                f"{d['runs']} runs, the median over interleaved rounds; shown as the median across runs with the "
                "range. All variants run in one process on one CUDA stream with their inputs on the device; each "
                "call is timed with CUDA events from before the call to the end of its last kernel, so host launch "
                f"overhead counts. The timer's own floor (a call that launches nothing) was {min(floors):.0f}–"
                f"{max(floors):.0f} µs. The SM clock column is the median sampled during each config: a laptop "
                "GPU stays near idle clocks when the calls are tiny. " + gpu_line(d))
    return "\n".join(rows)


TABLES = {
    "cpu-passes": cpu_passes,
    "cpu-torch": cpu_torch,
    "cpu-torch-tuned": lambda: cpu_torch("results/cpu/torch_compare_tuned_malloc.json"),
    "gpu-elementwise": gpu_elementwise,
    "gpu-matmul": gpu_matmul,
    "gpu-torch": gpu_torch,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--gpu", metavar="DIR", help="print only the GPU tables, from DIR (such as results/gpu)")
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    args = ap.parse_args()
    if args.gpu:  # relative to the repository root, or absolute
        for name, fn in [("gpu-elementwise", gpu_elementwise), ("gpu-matmul", gpu_matmul), ("gpu-torch", gpu_torch)]:
            print(f"## {name}\n\n{fn(args.gpu)}\n")
        return 0
    if args.print:
        for name, fn in TABLES.items():
            print(f"## {name}\n\n{fn()}\n")
        return 0
    readme = Path(args.readme)
    text = readme.read_text(encoding="utf-8")
    for name, fn in TABLES.items():
        pattern = re.compile(rf"(<!-- BEGIN {name} -->\n)(?:.*?\n)?(<!-- END {name} -->)", re.S)
        text, n = pattern.subn(lambda m: m.group(1) + fn() + "\n" + m.group(2), text)
        print(f"{'updated' if n else 'no markers for'} {name}")
    readme.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
