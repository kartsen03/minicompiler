#!/usr/bin/env python3
"""Combines repeated benchmark runs into one result file with ranges.

On a laptop the relative speed of memory-bound and compute-bound code depends
on the machine's state (which core type runs the code, what else uses memory
bandwidth), so the same comparison can give different ratios minutes apart.
The benchmark scripts therefore repeat each run and keep every raw file under
results/<device>/runs/. This script writes, for each configuration, the median
over runs of each run's headline ratio, with the minimum and maximum across
runs, plus the run whose ratio was the median (for its absolute timings).

    scripts/aggregate_results.py passes OUT.json RUN1.json RUN2.json ...
    scripts/aggregate_results.py torch OUT.json RUN1.json RUN2.json ...
"""

from __future__ import annotations

import json
import statistics
import sys


def spread(values: list[float]) -> dict:
    return {"median": statistics.median(values), "min": min(values), "max": max(values), "runs": values}


def aggregate_passes(runs: list[dict]) -> dict:
    out = {k: v for k, v in runs[0].items() if k != "results"}
    out["runs"] = len(runs)
    out["commits"] = sorted({r["environment"]["commit"] for r in runs})
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
    out = {k: v for k, v in runs[0].items() if k != "results"}
    out["runs"] = len(runs)
    out["commits"] = sorted({r["environment"]["commit"] for r in runs})
    results = []
    for i, first in enumerate(runs[0]["results"]):
        per_run = [r["results"][i] for r in runs]
        entry = {k: first[k] for k in ("graph", "size", "dims")}
        entry["minicompiler"] = {"median_ms": spread([p["minicompiler"]["timing"]["median_ms"] for p in per_run])}
        for v in ("eager", "eager_idiomatic", "compile"):
            entry[v] = {
                "median_ms": spread([p[v]["timing"]["median_ms"] for p in per_run]),
                "ratio_to_minicompiler": spread([p[v]["ratio_to_minicompiler"]["median"] for p in per_run]),
                "max_normwise_diff_vs_minicompiler": max(p[v]["max_normwise_diff_vs_minicompiler"] for p in per_run),
            }
        entry["compile"]["compile_seconds"] = spread([p["compile"]["compile_seconds"] for p in per_run])
        results.append(entry)
    out["results"] = results
    return out


def main() -> int:
    if len(sys.argv) < 4 or sys.argv[1] not in ("passes", "torch"):
        print(__doc__)
        return 2
    runs = []
    for path in sys.argv[3:]:
        with open(path, encoding="utf-8") as f:
            runs.append(json.load(f))
    combined = aggregate_passes(runs) if sys.argv[1] == "passes" else aggregate_torch(runs)
    with open(sys.argv[2], "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(combined, indent=2) + "\n")
    print(f"wrote {sys.argv[2]} from {len(runs)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
