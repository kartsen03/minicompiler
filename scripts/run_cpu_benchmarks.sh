#!/usr/bin/env bash
# Runs the CPU benchmarks and records them under results/cpu/:
#   passes.json         passes off vs on, Eigen backend (bench_cpu)
#   torch_compare.json  minicompiler vs PyTorch eager and torch.compile
#                       (needs a Python with torch; set PYTHON=...)
# Everything runs pinned to one core (CORE, default 2), single-threaded. Each
# result file records the commit, with -dirty if code was uncommitted.
#
#   scripts/run_cpu_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
CORE=${CORE:-2}
PYTHON=${PYTHON:-python3}

export MINICOMPILER_COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet -- . ':!results' || echo -dirty)"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" -j "$(nproc)" --target bench_cpu mcc > /dev/null
mkdir -p results/cpu

taskset -c "$CORE" "$BUILD_DIR/bench/bench_cpu" --out results/cpu/passes.json

if "$PYTHON" -c "import torch" 2> /dev/null; then
    taskset -c "$CORE" "$PYTHON" bench/torch_compare.py --mcc "$BUILD_DIR/tools/mcc" --device cpu \
        --out results/cpu/torch_compare.json
else
    echo "skipping the PyTorch comparison: '$PYTHON' cannot import torch" >&2
fi
