#!/usr/bin/env python3
"""Combines repeated benchmark runs into one result file with ranges.

On a laptop the relative speed of memory-bound and compute-bound code depends
on the machine's state (which core type runs the code, what else uses memory
bandwidth, how far the GPU boosts at its current temperature), so the same
comparison can give different ratios minutes apart. The benchmark scripts
therefore repeat each run and keep every raw file under results/<device>/runs/.
This script writes, for each configuration, the median over runs of each
run's headline numbers, with the minimum and maximum across runs.

    scripts/aggregate_results.py passes OUT.json RUN1.json RUN2.json ...
    scripts/aggregate_results.py torch OUT.json RUN1.json RUN2.json ...
    scripts/aggregate_results.py gpu_elementwise OUT.json RUN1.json ...
    scripts/aggregate_results.py gpu_matmul OUT.json RUN1.json ...
"""

from __future__ import annotations

import json
import statistics
import sys


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "runs": values}


def header(runs: list[dict]) -> dict:
    """Everything but the results from the first run, plus the run count, the
    commits, and the SM clock range over all runs for GPU results."""
    out = {k: v for k, v in runs[0].items() if k != "results"}
    out["runs"] = len(runs)
    out["commits"] = sorted({r["environment"]["commit"] for r in runs})
    clocks = [r["device"]["sm_clock_mhz_during_run"] for r in runs if "sm_clock_mhz_during_run" in r.get("device", {})]
    if clocks:
        out["device"] = dict(runs[0]["device"])
        out["device"]["sm_clock_mhz_during_run"] = {
            "min": min(c["min"] for c in clocks),
            "median_of_run_medians": statistics.median(c["median"] for c in clocks),
            "max": max(c["max"] for c in clocks),
        }
    return out


def aggregate_passes(runs: list[dict]) -> dict:
    out = header(runs)
    results = []
    for i, first in enumerate(runs[0]["results"]):
        per_run = [r["results"][i] for r in runs]
        entry = {k: first[k] for k in ("graph", "size", "dims")}
        variants = []
        for j, v in enumerate(first["variants"]):
            speedups = [p["variants"][j]["speedup_paired"]["median"] for p in per_run]
            medians = [p["variants"][j]["timing"]["median_ms"] for p in per_run]
            agg = {k: v[k] for k in v if k not in ("timing", "speedup_vs_none", "speedup_paired")}
            agg["speedup_paired"] = spread(speedups)
            agg["median_ms"] = spread(medians)
            variants.append(agg)
        entry["variants"] = variants
        results.append(entry)
    out["results"] = results
    return out


def aggregate_torch(runs: list[dict]) -> dict:
    out = header(runs)
    results = []
    for i, first in enumerate(runs[0]["results"]):
        per_run = [r["results"][i] for r in runs]
        entry = {k: first[k] for k in ("graph", "size", "dims")}
        entry["minicompiler"] = {"median_ms": spread([p["minicompiler"]["timing"]["median_ms"] for p in per_run])}
        # On the GPU: kernel launches and matmul kernels, the same in every run.
        entry["minicompiler"].update({k: v for k, v in first["minicompiler"].items() if k != "timing"})
        for k in ("sm_clock_mhz_median", "timer_floor_ms"):
            if k in first:
                entry[k] = spread([p[k] for p in per_run])
        for v in [k for k, x in first.items() if isinstance(x, dict) and "ratio_to_minicompiler" in x]:
            entry[v] = {
                "median_ms": spread([p[v]["timing"]["median_ms"] for p in per_run]),
                "ratio_to_minicompiler": spread([p[v]["ratio_to_minicompiler"]["median"] for p in per_run]),
                "max_normwise_diff_vs_minicompiler": max(p[v]["max_normwise_diff_vs_minicompiler"] for p in per_run),
            }
            if "compile_seconds" in first[v]:
                entry[v]["compile_seconds"] = spread([p[v]["compile_seconds"] for p in per_run])
        results.append(entry)
    out["results"] = results
    return out


def aggregate_gpu_elementwise(runs: list[dict]) -> dict:
    out = header(runs)
    results = []
    for i, first in enumerate(runs[0]["results"]):
        per_run = [r["results"][i] for r in runs]
        entry = {k: first[k] for k in ("size", "elements")}
        entry["max_normwise_diff_fused_vs_unfused"] = max(p["max_normwise_diff_fused_vs_unfused"] for p in per_run)
        variants = []
        for j, v in enumerate(first["variants"]):
            agg = {k: v[k] for k in ("variant", "passes", "kernel_launches", "bytes_moved")}
            agg["median_ms"] = spread([p["variants"][j]["timing"]["median_ms"] for p in per_run])
            for k in ("achieved_bandwidth_gbs", "percent_of_peak_bandwidth"):
                agg[k] = spread([p["variants"][j][k] for p in per_run])
            if "speedup_paired" in v:
                agg["speedup_paired"] = spread([p["variants"][j]["speedup_paired"]["median"] for p in per_run])
            variants.append(agg)
        entry["variants"] = variants
        results.append(entry)
    out["results"] = results
    return out


def aggregate_gpu_matmul(runs: list[dict]) -> dict:
    out = header(runs)
    results = []
    for i, first in enumerate(runs[0]["results"]):
        per_run = [r["results"][i] for r in runs]
        entry = {k: first[k] for k in ("shape", "m", "k", "n", "tiles_128x128", "register_tiled_picks")}
        entry["sm_clock_mhz_median"] = spread([p["sm_clock_mhz_median"] for p in per_run])
        # Which register tile size was faster in each run, against the pick.
        faster = []
        for p in per_run:
            by_name = {x["kernel"]: x for x in p["kernels"]}
            faster.append(max(("register_tiled_128", "register_tiled_64"), key=lambda n: by_name[n]["gflops"]))
        entry["faster_register_tile_per_run"] = faster
        kernels = []
        for j, kern in enumerate(first["kernels"]):
            agg = {"kernel": kern["kernel"]}
            agg["median_ms"] = spread([p["kernels"][j]["timing"]["median_ms"] for p in per_run])
            for k in ("gflops", "percent_of_peak_at_max_clock", "percent_of_peak_at_measured_clock",
                      "percent_of_cublas", "speedup_over_naive"):
                if k in kern:
                    agg[k] = spread([p["kernels"][j][k] for p in per_run])
            kernels.append(agg)
        entry["kernels"] = kernels
        results.append(entry)
    out["results"] = results
    return out


MODES = {"passes": aggregate_passes, "torch": aggregate_torch, "gpu_elementwise": aggregate_gpu_elementwise,
         "gpu_matmul": aggregate_gpu_matmul}


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] not in MODES:
        print(__doc__)
        return 2
    runs = []
    for path in sys.argv[3:]:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    combined = MODES[sys.argv[1]](runs)
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(combined, indent=2) + "\n")
    print(f"wrote {sys.argv[2]} from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
