#!/usr/bin/env python3
"""Summarizes the Nsight Compute reports written by scripts/profile_kernels_ncu.sh.

    scripts/ncu_summary.py NCU_DIR DOCS_DIR

NCU_DIR holds one `<label>.csv` per workload (`ncu --page raw --csv
--print-units base`), each with one row per kernel the workload launched.
Writes to DOCS_DIR:

  summary.md   the GELU chain's DRAM traffic unfused vs fused, and per matmul
               kernel: time, GFLOP/s, registers, occupancy, FMA-pipe use (or,
               for tensor cores, tensor throughput against the peak of the
               path used), shared-memory bank conflicts, cache hit rates,
               DRAM throughput and the top warp stall reasons
  metrics.csv  the same numbers, one row per workload
  roofline.svg every workload on a roofline: FLOP per DRAM byte against
               GFLOP/s, under the DRAM roof, the FP32 roof and the roof of each
               tensor path used (peaks as Nsight Compute reports them)
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

METRICS = {
    "time_ns": ["gpu__time_duration.sum"],
    "dram_read": ["dram__bytes_read.sum"],
    "dram_write": ["dram__bytes_write.sum"],
    "dram_pct": ["gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                 "dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    "sm_pct": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    "occ_achieved": ["sm__warps_active.avg.pct_of_peak_sustained_active"],
    "occ_theoretical": ["sm__maximum_warps_per_active_cycle_pct"],
    "registers": ["launch__registers_per_thread"],
    "block": ["launch__block_size"],
    "grid": ["launch__grid_size"],
    "fma_pct": ["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
                "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active"],
    "smem_ld_conflicts": ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"],
    "smem_ld_wavefronts": ["l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"],
    "l1_hit": ["l1tex__t_sector_hit_rate.pct"],
    "l2_hit": ["lts__t_sector_hit_rate.pct"],
    # Executed FP32 instructions per elapsed cycle, summed over the SM
    # sub-partitions; times the elapsed cycles, they are totals.
    "ffma_rate": ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed"],
    "fadd_rate": ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed"],
    "fmul_rate": ["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed"],
    "cycles": ["smsp__cycles_elapsed.avg"],
    "sm_hz": ["sm__cycles_elapsed.avg.per_second", "gpc__cycles_elapsed.avg.per_second"],
    "sm_count": ["device__attribute_multiprocessor_count"],
}
STALL = re.compile(r"smsp__average_warps_issue_stalled_(\w+?)_per_issue_active\.ratio$")
# Tensor-core work by input format, dense, accumulating in FP32: ops = FLOPs.
TENSOR_OPS = re.compile(r"sm__ops_path_tensor_src_(tf32|bf16|fp16)_dst_fp32_sparsity_off\.sum$")


def number(text: str) -> float:
    text = text.strip().replace(",", "")
    return float(text) if text not in ("", "n/a") else float("nan")


def read_kernels(path: Path) -> list[dict[str, str]]:
    """The kernel rows of a raw-page CSV, skipping the units row."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)]
    return [r for r in rows if r.get("ID", "").strip().isdigit()]


def pick(row: dict[str, str], key: str) -> float:
    for name in METRICS[key]:
        if name in row:
            return number(row[name])
    raise KeyError(f"no column for {key} ({METRICS[key]}); columns include {sorted(row)[:40]}")


def stalls(row: dict[str, str], top: int = 3) -> list[tuple[str, float]]:
    found = [(m.group(1), number(v)) for k, v in row.items() if (m := STALL.match(k)) and v.strip()]
    found = [(name, value) for name, value in found if not math.isnan(value) and name != "selected"]
    return sorted(found, key=lambda x: -x[1])[:top]


def summarize(label: str, kernels: list[dict[str, str]]) -> dict:
    """Totals over a workload's kernels; ratios from its longest kernel."""
    times = [pick(k, "time_ns") for k in kernels]
    main = kernels[times.index(max(times))]
    s = {"label": label, "kernels": len(kernels), "time_ns": sum(times),
         "dram_read": sum(pick(k, "dram_read") for k in kernels),
         "dram_write": sum(pick(k, "dram_write") for k in kernels),
         "flops": sum((2 * pick(k, "ffma_rate") + pick(k, "fadd_rate") + pick(k, "fmul_rate")) * pick(k, "cycles")
                      for k in kernels),
         "main_kernel": main.get("Kernel Name", "?"),
         # Time-weighted over the kernels, so nine unfused kernels average fairly.
         "dram_pct": sum(pick(k, "dram_pct") * t for k, t in zip(kernels, times)) / sum(times)}
    for key in ("sm_pct", "occ_achieved", "occ_theoretical", "registers", "block", "grid", "fma_pct",
                "smem_ld_conflicts", "smem_ld_wavefronts", "l1_hit", "l2_hit", "sm_hz", "sm_count"):
        s[key] = pick(main, key)
    s["stalls"] = stalls(main)
    # The tensor path the workload used most, its share of that path's peak
    # in the longest kernel, and the peak in ops per SM per clock.
    ops: dict[str, float] = {}
    for k in kernels:
        for name, value in k.items():
            if (m := TENSOR_OPS.match(name)) and value.strip():
                ops[m.group(1)] = ops.get(m.group(1), 0.0) + number(value)
    path = max(ops, key=ops.get) if ops and max(ops.values()) > 0 else ""
    s["tensor_path"] = path
    s["tensor_flops"] = ops.get(path, 0.0)
    if path:
        base = f"sm__ops_path_tensor_src_{path}_dst_fp32_sparsity_off"
        s["tensor_pct"] = number(main.get(f"{base}.sum.pct_of_peak_sustained_elapsed", ""))
        s["tensor_peak_per_sm_clock"] = number(main.get(f"{base}.avg.peak_sustained", ""))
    return s


def short(name: str) -> str:
    name = name.strip().removeprefix("void ")
    for cast in ("(int)", "(unsigned long)", "(bool)"):
        name = name.replace(cast, "")
    name = re.sub(r"\(.*$", "", name)
    name = re.sub(r"\b(minicompiler::cuda::|<unnamed>::|\(anonymous namespace\)::)", "", name)
    return name if len(name) <= 60 else name[:57] + "..."


def fmt_bytes(b: float) -> str:
    return f"{b / 2**20:,.0f} MiB" if b >= 2**20 else f"{b / 2**10:,.0f} KiB"


def device_peaks() -> tuple[float, int, int]:
    """Peak DRAM bandwidth and the FP32 lanes, from the recorded GPU results."""
    d = json.loads((ROOT / "results/gpu/matmul.json").read_text(encoding="utf-8"))["device"]
    return d["peak_bandwidth_gbs"], d["sm_count"], d["fp32_lanes_per_sm"]


def roofline_svg(points: list[tuple[str, float, float]], peak_bw: float, peak_flops: float,
                 tensor_roofs: dict[str, float]) -> str:
    """A log-log roofline: x = FLOP per DRAM byte, y = GFLOP/s, with a dashed
    roof per tensor path (input format) at its peak."""
    w, h, left, bottom, top, right = 760, 460, 70, 50, 20, 20
    x0, x1, y0, y1 = 0.05, 2000.0, 10.0, 30000.0

    def px(x: float) -> float:
        return left + (math.log10(x) - math.log10(x0)) / (math.log10(x1) - math.log10(x0)) * (w - left - right)

    def py(y: float) -> float:
        return h - bottom - (math.log10(y) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)) * (h - top - bottom)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" '
           f'font-size="11"><rect width="100%" height="100%" fill="white"/>']
    for dec in range(-1, 4):
        x = 10.0 ** dec
        out.append(f'<line x1="{px(x):.1f}" y1="{top}" x2="{px(x):.1f}" y2="{h - bottom}" stroke="#ddd"/>')
        out.append(f'<text x="{px(x):.1f}" y="{h - bottom + 16}" text-anchor="middle">{x:g}</text>')
    for dec in range(1, 5):
        y = 10.0 ** dec
        out.append(f'<line x1="{left}" y1="{py(y):.1f}" x2="{w - right}" y2="{py(y):.1f}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 6}" y="{py(y) + 4:.1f}" text-anchor="end">{y:g}</text>')
    knee = peak_flops / peak_bw
    out.append(f'<polyline fill="none" stroke="#333" stroke-width="2" points="{px(x0):.1f},{py(max(y0, peak_bw * x0)):.1f} '
               f'{px(knee):.1f},{py(peak_flops):.1f} {px(x1):.1f},{py(peak_flops):.1f}"/>')
    out.append(f'<text x="{px(x1) - 4:.1f}" y="{py(peak_flops) - 6:.1f}" text-anchor="end">FP32 peak at the profiling '
               f'clock: {peak_flops:,.0f} GFLOP/s</text>')
    out.append(f'<text x="{px(0.12):.1f}" y="{py(peak_bw * 0.12) - 8:.1f}">DRAM {peak_bw:.0f} GB/s</text>')
    # Paths with the same peak share a roof and a label.
    by_peak: dict[int, list[str]] = {}
    for path, roof in tensor_roofs.items():
        by_peak.setdefault(round(roof), []).append(path.upper())
    for roof, paths in sorted(by_peak.items()):
        knee_t = roof / peak_bw
        out.append(f'<polyline fill="none" stroke="#333" stroke-width="1.5" stroke-dasharray="6 4" '
                   f'points="{px(knee_t):.1f},{py(roof):.1f} {px(x1):.1f},{py(roof):.1f}"/>')
        out.append(f'<text x="{px(x1) - 4:.1f}" y="{py(roof) - 6:.1f}" text-anchor="end">'
                   f'{" and ".join(sorted(paths))} tensor peak: {roof:,.0f} GFLOP/s</text>')
    colors = {"naive": "#999", "tiled": "#e69f00", "register_tiled_128": "#0072b2", "register_tiled_64": "#56b4e9",
              "vectorized": "#332288", "double_buffered": "#117733", "split_k": "#44aa99", "cublas": "#d55e00",
              "tensor_core_tf32": "#88ccee", "tensor_core_bf16": "#ddcc77", "tensor_core_f16": "#aa4499",
              "cublas_tf32": "#882255", "cublas_bf16": "#999933", "cublas_f16": "#661100",
              "unfused": "#cc79a7", "fused": "#009e73"}
    for label, ai, gflops in points:
        kernel = label.split(".")[-1]
        color = colors.get(kernel, "#000")
        out.append(f'<circle cx="{px(ai):.1f}" cy="{py(gflops):.1f}" r="4" fill="{color}"><title>{label}: '
                   f'{ai:.2f} FLOP/B, {gflops:,.0f} GFLOP/s</title></circle>')
    y = top + 14
    for kernel, color in colors.items():
        out.append(f'<circle cx="{left + 14}" cy="{y - 4}" r="4" fill="{color}"/>'
                   f'<text x="{left + 24}" y="{y}">{kernel}</text>')
        y += 15
    out.append(f'<text x="{(left + w - right) / 2:.0f}" y="{h - 12}" text-anchor="middle">FLOP per DRAM byte</text>')
    out.append(f'<text x="16" y="{(top + h - bottom) / 2:.0f}" transform="rotate(-90 16 {(top + h - bottom) / 2:.0f})" '
               'text-anchor="middle">GFLOP/s</text></svg>')
    return "\n".join(out) + "\n"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    ncu_dir, docs = Path(sys.argv[1]), Path(sys.argv[2])
    docs.mkdir(parents=True, exist_ok=True)
    rows = [summarize(p.stem, read_kernels(p)) for p in sorted(ncu_dir.glob("*.csv"))]
    if not rows:
        print(f"no reports in {ncu_dir}")
        return 1
    peak_bw, sms, lanes = device_peaks()
    clocks = sorted(r["sm_hz"] for r in rows)
    sm_ghz = clocks[len(clocks) // 2] / 1e9
    peak_flops = 2.0 * sms * lanes * sm_ghz  # GFLOP/s at the clock ncu ran at

    out = ["# Nsight Compute summary", "",
           f"One launch of each kernel, full metric set, the GPU held at {sm_ghz * 1000:.0f} MHz by Nsight Compute "
           f"(FP32 peak there: {peak_flops:,.0f} GFLOP/s). See `README.md` here for what the numbers show.", ""]
    gelu = {r["label"].split(".")[1]: r for r in rows if r["label"].startswith("gelu_")}
    if gelu:
        input_bytes = 4096 * 4096 * 4
        out += ["## GELU chain at 64 MB", "",
                "| Variant | Kernels | Time | DRAM read | DRAM written | Traffic ÷ input size | DRAM throughput |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for name in ("unfused", "fused"):
            r = gelu.get(name)
            if r:
                traffic = (r["dram_read"] + r["dram_write"]) / input_bytes
                out.append(f"| {name} | {r['kernels']} | {r['time_ns'] / 1e3:,.0f} µs | {fmt_bytes(r['dram_read'])} | "
                           f"{fmt_bytes(r['dram_write'])} | {traffic:.1f} | {r['dram_pct']:.0f}% |")
        out.append("")
    shapes = []
    for r in rows:
        if r["label"].startswith("matmul_"):
            shape = r["label"].split(".")[0].removeprefix("matmul_")
            if shape not in shapes:
                shapes.append(shape)
    fp32_order = ["naive", "tiled", "register_tiled_128", "register_tiled_64", "vectorized", "double_buffered",
                  "split_k", "cublas"]
    tensor_order = ["tensor_core_tf32", "cublas_tf32", "tensor_core_bf16", "cublas_bf16", "tensor_core_f16",
                    "cublas_f16"]
    for shape in shapes:
        m, k, n = (int(x) for x in shape.split("x"))
        by_kernel = {r["label"].split(".")[1]: r for r in rows if r["label"].startswith(f"matmul_{shape}.")}

        def common(name: str, r: dict) -> tuple[str, str, str]:
            conflicts = (f"{r['smem_ld_conflicts'] / r['smem_ld_wavefronts']:.0%} of wavefronts"
                         if r["smem_ld_wavefronts"] > 0 else "no shared loads")
            st = ", ".join(f"{s} {v:.1f}" for s, v in r["stalls"])
            label = name if not name.startswith("cublas") else f"{name} (`{short(r['main_kernel'])}`)"
            return label, conflicts, st

        out += [f"## Matmul {shape} (m × k × n)", "",
                "| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | FMA pipe | "
                "Shared-load bank conflicts | L1 hit | L2 hit | DRAM | Top stall reasons (cycles per issue) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name in fp32_order:
            r = by_kernel.get(name)
            if not r:
                continue
            gflops = 2.0 * m * n * k / r["time_ns"]
            label, conflicts, st = common(name, r)
            out.append(f"| {label} | {r['time_ns'] / 1e3:,.1f} µs | {gflops:,.0f} | {r['registers']:.0f} | "
                       f"{r['block']:.0f} × {r['grid']:.0f} | {r['occ_achieved']:.0f}% / {r['occ_theoretical']:.0f}% | "
                       f"{r['fma_pct']:.0f}% | {conflicts} | {r['l1_hit']:.0f}% | {r['l2_hit']:.0f}% | "
                       f"{r['dram_pct']:.0f}% | {st} |")
        out.append("")
        tensor = [(name, by_kernel[name]) for name in tensor_order if name in by_kernel]
        if tensor:
            out += [f"Tensor cores on {shape}: the inputs rounded to each format, FP32 accumulation. Tensor "
                    "throughput is the tensor ops executed against the peak of the path used, as Nsight Compute "
                    "reports both.", "",
                    "| Kernel | Time | GFLOP/s | Regs | Block × grid | Occupancy achieved / theoretical | "
                    "Tensor throughput (path) | Shared-load bank conflicts | L2 hit | DRAM | "
                    "Top stall reasons (cycles per issue) |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
            for name, r in tensor:
                gflops = 2.0 * m * n * k / r["time_ns"]
                label, conflicts, st = common(name, r)
                path = (f"{r['tensor_pct']:.0f}% ({r['tensor_path'].upper()} → FP32)" if r["tensor_path"]
                        else "no tensor ops")
                out.append(f"| {label} | {r['time_ns'] / 1e3:,.1f} µs | {gflops:,.0f} | {r['registers']:.0f} | "
                           f"{r['block']:.0f} × {r['grid']:.0f} | {r['occ_achieved']:.0f}% / "
                           f"{r['occ_theoretical']:.0f}% | {path} | {conflicts} | {r['l2_hit']:.0f}% | "
                           f"{r['dram_pct']:.0f}% | {st} |")
            out.append("")
    (docs / "summary.md").write_text("\n".join(out), encoding="utf-8", newline="\n")

    fields = ["label", "kernels", "main_kernel", "time_ns", "flops", "dram_read", "dram_write", "dram_pct", "sm_pct",
              "occ_achieved", "occ_theoretical", "registers", "block", "grid", "fma_pct", "smem_ld_conflicts",
              "smem_ld_wavefronts", "l1_hit", "l2_hit", "sm_hz", "tensor_path", "tensor_flops", "tensor_pct",
              "tensor_peak_per_sm_clock"]
    with open(docs / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["stalls"], extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "stalls": "; ".join(f"{s}={v:.2f}" for s, v in r["stalls"])})

    # Matmuls count useful work, 2mnk (the executed FMAs also include the
    # zero-padded edge tiles); the GELU chain counts its executed FP32 ops.
    points = []
    for r in rows:
        dram = r["dram_read"] + r["dram_write"]
        flops = r["flops"]
        if r["label"].startswith("matmul_"):
            m, k, n = (int(x) for x in r["label"].split(".")[0].removeprefix("matmul_").split("x"))
            flops = 2.0 * m * n * k
        if flops > 0 and dram > 0:
            points.append((r["label"], flops / dram, flops / r["time_ns"]))
    # The roof of each tensor path used: its peak ops per SM per clock (from
    # Nsight Compute) x SMs x the profiling clock.
    tensor_roofs: dict[str, float] = {}
    for r in rows:
        if r["tensor_path"] and r.get("tensor_peak_per_sm_clock", 0) > 0:
            tensor_roofs[r["tensor_path"]] = r["tensor_peak_per_sm_clock"] * sms * sm_ghz
    (docs / "roofline.svg").write_text(roofline_svg(points, peak_bw, peak_flops, tensor_roofs), encoding="utf-8",
                                       newline="\n")
    print(f"wrote {docs}/summary.md, metrics.csv, roofline.svg ({len(rows)} workloads)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
