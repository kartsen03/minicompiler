#!/usr/bin/env python3
"""Checks that minicompiler's timings agree across benchmark runs.

bench_cpu (passes.json) and the PyTorch comparisons each time minicompiler with
all passes on the same configurations. If the medians disagree by more than
25%, something else was competing for the core during one of the runs, and the
results should not be recorded.

    scripts/check_timing_consistency.py results/cpu/passes.json results/cpu/torch_compare*.json
"""

import json
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    with open(sys.argv[1], encoding="utf-8") as f:
        passes = json.load(f)
    reference = {}
    for r in passes["results"]:
        full = next(v for v in r["variants"] if v["variant"] == "all")
        reference[(r["graph"], r["size"])] = full["timing"]["median_ms"]

    ok = True
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as f:
            runs = json.load(f)["results"]
        for r in runs:
            ratio = r["minicompiler"]["timing"]["median_ms"] / reference[(r["graph"], r["size"])]
            consistent = 0.8 <= ratio <= 1.25
            ok &= consistent
            print(f"{path}: {r['graph']:17} {r['size']:24} {ratio:5.2f}{'' if consistent else '  <-- inconsistent'}")
    print("timings consistent" if ok else "TIMINGS INCONSISTENT: rerun on an idle machine")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
