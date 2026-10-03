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


def env_line(env: dict, keys: list[tuple[str, str]]) -> str:
    parts = [f"{label} {env[k]}" for k, label in keys if k in env]
    return "Measured on " + ", ".join(parts) + "."


def cpu_passes() -> str:
    d = load("results/cpu/passes.json")
    if not d:
        return "_No results recorded yet._"
    rows = ["| Graph | Size | Nodes | Compute nodes | Passes off | Passes on | Speedup |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for r in d["results"]:
        v = {x["variant"]: x for x in r["variants"]}
        none, full = v["none"], v["all"]
        rows.append(f"| `{r['graph']}` | {r['size']} | {none['nodes']} → {full['nodes']} | "
                    f"{none['compute_nodes']} → {full['compute_nodes']} | {ms(none['timing']['median_ms'])} ms | "
                    f"{ms(full['timing']['median_ms'])} ms | {full['speedup_vs_none']:.2f}x |")
    e = d["environment"]
    rows.append("")
    rows.append(env_line(e, [("cpu", "CPU:"), ("os", "OS:"), ("compiler", "compiler:"), ("eigen", "Eigen"),
                             ("build_flags", "build:"), ("threads", "threads:"),
                             ("host_power_plan", "Windows power plan:"), ("commit", "commit")])
                + " Median of interleaved runs; see `results/cpu/passes.json` for p10/p90 and the per-pass ablation.")
    return "\n".join(rows)


def cpu_torch(path: str = "results/cpu/torch_compare.json") -> str:
    d = load(path)
    if not d:
        return "_Not recorded yet._"
    rows = ["| Graph | Size | minicompiler | PyTorch eager (same ops) | PyTorch eager (idiomatic) | torch.compile |",
            "|---|---|---:|---:|---:|---:|"]
    for r in d["results"]:
        def cell(key: str) -> str:
            t = r[key]["timing"]["median_ms"]
            mc = r["minicompiler"]["timing"]["median_ms"]
            if key == "minicompiler":
                return f"{ms(t)} ms"
            return f"{ms(t)} ms ({t / mc:.2f}x)"
        rows.append(f"| `{r['graph']}` | {r['size']} | {cell('minicompiler')} | {cell('eager')} | "
                    f"{cell('eager_idiomatic')} | {cell('compile')} |")
    e = d["environment"]
    rows.append("")
    rows.append("In parentheses: PyTorch time divided by minicompiler time (above 1 means minicompiler is faster). "
                + env_line(e, [("cpu", "CPU:"), ("torch", "PyTorch"), ("python", "Python"),
                               ("torch_threads", "threads:"), ("malloc_mmap_threshold", "glibc mmap threshold:"),
                               ("host_power_plan", "Windows power plan:"), ("commit", "commit")]))
    return "\n".join(rows)


TABLES = {
    "cpu-passes": cpu_passes,
    "cpu-torch": cpu_torch,
    "cpu-torch-tuned": lambda: cpu_torch("results/cpu/torch_compare_tuned_malloc.json"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--print", action="store_true")
    ap.add_argument("--readme", default=str(ROOT / "README.md"))
    args = ap.parse_args()
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
