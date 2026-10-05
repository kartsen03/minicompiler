#!/usr/bin/env python3
"""Summarizes the Nsight Systems reports written by scripts/profile_gpu.sh.

    scripts/nsys_summary.py STATS_DIR DOCS_DIR

Reads stats_nvtx_sum.csv, stats_nvtx_gpu_proj_sum.csv and
stats_nvtx_kern_sum.csv from STATS_DIR. Each NVTX range is one synchronized
call of one variant ("<config>/<variant>"), so for every variant it reports
the call's wall time, the time the GPU spent on the work the call launched
(projected onto the range), and the kernels behind it. Writes summary.md and
copies the three CSV files into DOCS_DIR.
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

VARIANTS = ["minicompiler", "eager", "eager_idiomatic", "compile", "compile_cudagraphs"]


def rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def col(row: dict, *names: str) -> str:
    """The first column whose header matches one of `names`, ignoring case."""
    lowered = {k.strip().lower(): v for k, v in row.items() if k}
    for n in names:
        if n.lower() in lowered:
            return lowered[n.lower()]
    raise KeyError(f"none of {names} in {list(row)}")


def range_name(text: str) -> str:
    return text.strip().lstrip(":")


def us(ns: str | float) -> float:
    return float(ns) / 1000.0


def short_kernel(name: str) -> str:
    """The kernel's name without its return type, parameter list, casts in
    its template arguments and the usual namespaces."""
    name = name.strip().removeprefix("void ")
    for cast in ("(int)", "(unsigned long)", "(bool)"):
        name = name.replace(cast, "")
    if name.endswith(")"):  # drop the parameter list: the last top-level (...)
        depth = 0
        for i in range(len(name) - 1, -1, -1):
            depth += {")": 1, "(": -1}.get(name[i], 0)
            if depth == 0:
                name = name[:i]
                break
    name = re.sub(r"\b(at::native::|minicompiler::cuda::|std::)|<unnamed>::", "", name)
    return name if len(name) <= 80 else name[:77] + "..."


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    stats, docs = Path(sys.argv[1]), Path(sys.argv[2])
    docs.mkdir(parents=True, exist_ok=True)
    files = {k: stats / f"stats_{k}.csv" for k in ("nvtx_sum", "nvtx_gpu_proj_sum", "nvtx_kern_sum")}

    wall = {}  # range -> (instances, median wall ns)
    for r in rows(files["nvtx_sum"]):
        wall[range_name(col(r, "Range"))] = (int(col(r, "Instances")), float(col(r, "Med (ns)")))
    proj = {}  # range -> (median projected GPU ns, average GPU ops per call)
    for r in rows(files["nvtx_gpu_proj_sum"]):
        proj[range_name(col(r, "Range"))] = (float(col(r, "Proj Med (ns)")), float(col(r, "Avg GPU Ops")))
    kernels = defaultdict(list)  # range -> [(kernel, instances per call, avg ns)]
    for r in rows(files["nvtx_kern_sum"]):
        name = range_name(col(r, "NVTX Range"))
        per_call = int(col(r, "Kern Inst")) / max(1, int(col(r, "NVTX Inst")))
        kernels[name].append((short_kernel(col(r, "Kernel Name")), per_call, float(col(r, "Avg (ns)"))))

    configs = []
    for name in wall:
        config = name.split("/")[0]
        if config not in configs:
            configs.append(config)
    out = ["# Nsight Systems summary", "",
           "One NVTX range per synchronized call (see `bench/profile_gpu.py`). *Wall* is the range's median "
           "duration on the host: Python, launches and the wait for the GPU. *GPU span* is the median time from "
           "the start of the call's first GPU operation to the end of its last (Nsight's projection of the "
           "range onto the GPU), gaps included. *Kernel time* is the sum over the call's kernels of their "
           "average durations. Wall minus kernel time is the time the call spent outside kernels.", ""]
    for config in configs:
        out += [f"## {config}", "",
                "| Variant | Calls | Wall | GPU span | Kernel time | Kernels per call | Kernels (per call × average) |",
                "|---|---:|---:|---:|---:|---:|---|"]
        for v in VARIANTS:
            key = f"{config}/{v}"
            if key not in wall:
                continue
            calls, wall_ns = wall[key]
            span_ns, _ = proj.get(key, (0.0, 0.0))
            ks = sorted(kernels.get(key, []), key=lambda k: -k[1] * k[2])
            kernel_ns = sum(n * t for _, n, t in ks)
            listed = "; ".join(f"`{k}` {n:g} × {us(t):.1f} µs" for k, n, t in ks[:4])
            if len(ks) > 4:
                listed += f"; {len(ks) - 4} more"
            out.append(f"| {v} | {calls} | {us(wall_ns):.1f} µs | {us(span_ns):.1f} µs | {us(kernel_ns):.1f} µs | "
                       f"{sum(n for _, n, _ in ks):g} | {listed} |")
        out.append("")
    (docs / "summary.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    for path in files.values():
        shutil.copy(path, docs / path.name)
    print(f"wrote {docs / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
