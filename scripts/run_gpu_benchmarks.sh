#!/usr/bin/env bash
# Runs the GPU benchmarks and records them under results/gpu/:
#   elementwise.json    the GELU chain as one kernel per op vs one fused
#                       kernel, with bandwidth against the GPU's peak
#   matmul.json         the naive, tiled and register-tiled matmul kernels
#                       against cuBLAS, in GFLOP/s and percent of peak FP32
#   torch_compare.json  minicompiler's CUDA backend against PyTorch eager and
#                       torch.compile on the same GPU and the same stream
# Needs a build with the CUDA backend (BUILD_DIR, default build) and, for the
# PyTorch comparison, a Python with a CUDA build of torch (set PYTHON=...).
#
# Each benchmark interleaves its variants within one process and reports the
# median of per-round ratios, which cancels slow drifts that hit every variant
# alike. A laptop GPU's clock still moves with power and temperature from one
# run to the next, so the whole sequence is repeated RUNS times (default 3);
# every raw run is kept under results/gpu/runs/ and the summaries give the
# median across runs with the min-max range. Each file records the commit
# (with -dirty if code was uncommitted), the host power plan and the SM clocks
# sampled during the run.
#
#   scripts/run_gpu_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
PYTHON=${PYTHON:-python3}
RUNS=${RUNS:-3}

export MINICOMPILER_COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet -- . ':!results' || echo -dirty)"
# Under WSL, record the Windows power plan: laptop power modes cap clocks.
if command -v powershell.exe > /dev/null; then
    MINICOMPILER_HOST_POWER_PLAN=$(powershell.exe -NoProfile -Command "powercfg /getactivescheme" 2> /dev/null |
        tr -d '\r' | sed -nE 's/.*\((.*)\).*/\1/p')
    export MINICOMPILER_HOST_POWER_PLAN
fi
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release > /dev/null
if ! cmake --build "$BUILD_DIR" -j "$(nproc)" --target bench_cuda minicompiler_capi > /dev/null; then
    echo "could not build bench_cuda: is '$BUILD_DIR' configured with the CUDA backend?" >&2
    exit 1
fi

with_torch=1
"$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2> /dev/null || {
    with_torch=0
    echo "skipping the PyTorch comparison: '$PYTHON' has no torch with CUDA" >&2
}

RAW=results/gpu/runs
rm -rf "$RAW"
mkdir -p "$RAW"
for run in $(seq 1 "$RUNS"); do
    echo "=== run $run of $RUNS"
    "$BUILD_DIR/bench/bench_cuda" --suite elementwise --out "$RAW/elementwise_$run.json"
    "$BUILD_DIR/bench/bench_cuda" --suite matmul --out "$RAW/matmul_$run.json"
    if [ "$with_torch" = 1 ]; then
        "$PYTHON" bench/torch_compare.py --device cuda --capi "$BUILD_DIR/bench/libminicompiler_capi.so" \
            --out "$RAW/torch_compare_$run.json"
    fi
done

"$PYTHON" scripts/aggregate_results.py gpu_elementwise results/gpu/elementwise.json "$RAW"/elementwise_*.json
"$PYTHON" scripts/aggregate_results.py gpu_matmul results/gpu/matmul.json "$RAW"/matmul_*.json
if [ "$with_torch" = 1 ]; then
    "$PYTHON" scripts/aggregate_results.py torch results/gpu/torch_compare.json "$RAW"/torch_compare_*.json
fi
