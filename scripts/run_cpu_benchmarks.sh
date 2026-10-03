#!/usr/bin/env bash
# Runs the CPU benchmarks and records them under results/cpu/:
#   passes.json                      passes off vs on, Eigen backend (bench_cpu)
#   torch_compare.json               minicompiler vs PyTorch eager and
#                                    torch.compile, PyTorch out of the box
#   torch_compare_tuned_malloc.json  the same with glibc keeping large freed
#                                    blocks in the heap, which removes the page
#                                    faults PyTorch pays for every fresh output
#                                    tensor (minicompiler reuses its buffers)
# The PyTorch runs need a Python with torch (set PYTHON=...).
#
# Everything runs pinned to one core (CORE, default 2), single-threaded. Each
# result file records the commit, with -dirty if code was uncommitted. The run
# fails if minicompiler's timings inside the PyTorch runs drift more than 25%
# from bench_cpu's: that means something else was competing for the core.
#
#   scripts/run_cpu_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
CORE=${CORE:-2}
PYTHON=${PYTHON:-python3}

export MINICOMPILER_COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet -- . ':!results' || echo -dirty)"
# Under WSL, record the Windows power plan: laptop power modes cap clocks.
if command -v powershell.exe > /dev/null; then
    MINICOMPILER_HOST_POWER_PLAN=$(powershell.exe -NoProfile -Command "powercfg /getactivescheme" 2> /dev/null |
        tr -d '\r' | sed -nE 's/.*\((.*)\).*/\1/p')
    export MINICOMPILER_HOST_POWER_PLAN
fi
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build "$BUILD_DIR" -j "$(nproc)" --target bench_cpu mcc > /dev/null
mkdir -p results/cpu

taskset -c "$CORE" "$BUILD_DIR/bench/bench_cpu" --out results/cpu/passes.json

if ! "$PYTHON" -c "import torch" 2> /dev/null; then
    echo "skipping the PyTorch comparison: '$PYTHON' cannot import torch" >&2
    exit 0
fi
compare() {  # output file
    taskset -c "$CORE" "$PYTHON" bench/torch_compare.py --mcc "$BUILD_DIR/tools/mcc" --device cpu --out "$1"
}
compare results/cpu/torch_compare.json
MALLOC_MMAP_THRESHOLD_=4294967296 MALLOC_TRIM_THRESHOLD_=68719476736 compare results/cpu/torch_compare_tuned_malloc.json
"$PYTHON" scripts/check_timing_consistency.py results/cpu/passes.json \
    results/cpu/torch_compare.json results/cpu/torch_compare_tuned_malloc.json
