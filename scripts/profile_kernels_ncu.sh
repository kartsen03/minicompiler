#!/usr/bin/env bash
# Collects Nsight Compute's full metric set for one launch of every GPU kernel
# and summarizes it in docs/profiling/gpu/ncu/: the GELU chain at 64 MB
# unfused and fused, and every matmul kernel plus cuBLAS on three shapes
# (bench_cuda --suite profile --list). Each workload is profiled in its own
# ncu run, filtered to its NVTX range, so a report holds exactly that
# workload's kernels.
#
# Needs ncu with access to the GPU's performance counters (otherwise it stops
# with ERR_NVGPUCTRPERM), a build with the CUDA backend (BUILD_DIR, default
# build) and Python 3. Nsight Compute locks the GPU to its base clock while
# profiling (--clock-control base, the default), so durations here are not the
# benchmark's; utilization percentages are relative to that clock. The
# .ncu-rep files stay in OUT_DIR (default ~/.cache/minicompiler-ncu, not /tmp,
# which WSL wipes when the distribution stops) and can be opened in the Nsight
# Compute GUI; they are not committed.
#
# A hot GPU can slow below the locked clock, and ncu_summary.py then names the
# workloads that ran slow. LABELS="label ..." profiles just those again,
# keeping every other report; each report records the commit it was taken at,
# and environment.txt lists them.
#
#   scripts/profile_kernels_ncu.sh
#   LABELS="matmul_2048x2048x2048.split_k" scripts/profile_kernels_ncu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
PYTHON=${PYTHON:-python3}
OUT_DIR=${OUT_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/minicompiler-ncu}
DOCS=docs/profiling/gpu/ncu

cmake --build "$BUILD_DIR" -j "$(nproc)" --target bench_cuda > /dev/null
BENCH="$BUILD_DIR/bench/bench_cuda"
COMMIT="$(git rev-parse --short HEAD)$(git diff --quiet -- . ':!results' ':!docs' || echo -dirty)"
if [ -z "${LABELS:-}" ]; then
    rm -rf "$OUT_DIR"
    LABELS=$("$BENCH" --suite profile --list)
fi
mkdir -p "$OUT_DIR" "$DOCS"
for label in $LABELS; do
    echo "profiling $label"
    echo "$COMMIT" > "$OUT_DIR/$label.commit"
    ncu --set full --nvtx --nvtx-include "$label/" --force-overwrite --export "$OUT_DIR/$label" \
        "$BENCH" --suite profile --only "$label" > "$OUT_DIR/$label.log" 2>&1 || {
        tail -20 "$OUT_DIR/$label.log"
        exit 1
    }
    ncu --import "$OUT_DIR/$label.ncu-rep" --page raw --csv --print-units base > "$OUT_DIR/$label.csv"
done
"$PYTHON" scripts/ncu_summary.py "$OUT_DIR" "$DOCS"
{
    echo "ncu: $(ncu --version | tail -1)"
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
    echo "commit: $(sort -u "$OUT_DIR"/*.commit | paste -sd ' ')"
} > "$DOCS/environment.txt"
echo "wrote $DOCS"
