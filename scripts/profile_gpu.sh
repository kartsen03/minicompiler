#!/usr/bin/env bash
# Profiles the GPU comparison with Nsight Systems and writes text summaries to
# docs/profiling/gpu/. bench/profile_gpu.py calls each variant (minicompiler's
# CUDA backend, PyTorch eager, idiomatic eager, torch.compile with and without
# CUDA graphs) in synchronized calls, each in an NVTX range, so the summaries
# split every call's wall time into the GPU kernels it ran and the rest.
# Needs nsys, a build with the CUDA backend (BUILD_DIR, default build) and a
# Python with a CUDA build of torch (PYTHON). The .nsys-rep file is kept in
# OUT_DIR (default ~/.cache/minicompiler-nsys); it is large and not committed.
#
#   scripts/profile_gpu.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD_DIR=${BUILD_DIR:-build}
PYTHON=${PYTHON:-python3}
OUT_DIR=${OUT_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/minicompiler-nsys}
DOCS=docs/profiling/gpu

cmake --build "$BUILD_DIR" -j "$(nproc)" --target minicompiler_capi > /dev/null
mkdir -p "$OUT_DIR" "$DOCS"
# --cuda-graph-trace=node traces the kernels inside torch.compile's CUDA
# graphs one by one instead of the graph as a single item.
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true \
    -o "$OUT_DIR/compare" \
    "$PYTHON" bench/profile_gpu.py --capi "$BUILD_DIR/bench/libminicompiler_capi.so" > "$OUT_DIR/profile.log" 2>&1 || {
    tail -20 "$OUT_DIR/profile.log"
    exit 1
}
# Per NVTX range (one per config and variant): the calls' wall time, the GPU
# time projected onto them, and the kernels each variant launched.
for report in nvtx_sum nvtx_gpu_proj_sum nvtx_kern_sum; do
    nsys stats --force-export=true --report "$report" --format csv --output "$OUT_DIR/stats" \
        "$OUT_DIR/compare.nsys-rep" > /dev/null
done
"$PYTHON" scripts/nsys_summary.py "$OUT_DIR" "$DOCS"
{
    echo "nsys: $(nsys --version)"
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader)"
    echo "commit: $(git rev-parse --short HEAD)$(git diff --quiet -- . ':!results' ':!docs' || echo -dirty)"
} > "$DOCS/environment.txt"
echo "wrote $DOCS"
