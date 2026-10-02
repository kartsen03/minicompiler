#!/usr/bin/env bash
# Renders every benchmark graph before and after each pass of the default
# pipeline (dne, fold, dne, fuse, dne) into docs/ir_diagrams/<graph>/:
# NN_<stage>.dot from `mcc --dump-dot`, and NN_<stage>.svg from Graphviz.
#
#   scripts/render_ir_diagrams.sh [path/to/mcc]
set -euo pipefail
cd "$(dirname "$0")/.."
MCC=${1:-build/tools/mcc}
command -v dot >/dev/null || { echo "Graphviz 'dot' is required" >&2; exit 1; }

for graph in gelu_chain matmul_bias_relu mlp_block; do
    out=docs/ir_diagrams/$graph
    rm -rf "$out"
    mkdir -p "$out"
    "$MCC" "bench/graphs/$graph.mcg" --dump-dot "$out"
    for f in "$out"/*.dot; do
        dot -Tsvg "$f" -o "${f%.dot}.svg"
    done
    echo "$out: $(ls "$out" | grep -c '\.svg$') diagrams"
done
