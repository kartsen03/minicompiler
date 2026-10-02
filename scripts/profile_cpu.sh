#!/usr/bin/env bash
# Profiles mcc with Linux perf and renders flame graphs into docs/profiling/.
#
# Builds a Release tree with frame pointers and debug info, samples with
# `perf record -e cpu-clock -g` (a software timer event, so it also works in
# VMs and WSL2 where hardware counters are unavailable), and folds the stacks
# with Brendan Gregg's FlameGraph scripts.
#
#   scripts/profile_cpu.sh
#
# Environment: PERF (perf binary), FLAMEGRAPH_DIR (FlameGraph checkout),
# BUILD_DIR (profiling build tree).
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
BUILD_DIR=${BUILD_DIR:-$HOME/mc/build-prof}
FLAMEGRAPH_DIR=${FLAMEGRAPH_DIR:-$HOME/mc/FlameGraph}
OUT=docs/profiling

# On WSL2 and some cloud kernels /usr/bin/perf is a wrapper that refuses to
# run when no linux-tools package matches the running kernel; the versioned
# binary works.
if [ -z "${PERF:-}" ]; then
    PERF=$(ls /usr/lib/linux-tools/*/perf 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$PERF" ] || PERF=perf
fi
"$PERF" --version >/dev/null

if [ ! -d "$FLAMEGRAPH_DIR" ]; then
    git clone --depth 1 https://github.com/brendangregg/FlameGraph.git "$FLAMEGRAPH_DIR"
fi

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DMINICOMPILER_FRAME_POINTERS=ON \
      -DCMAKE_CXX_FLAGS="-g" -DMINICOMPILER_BUILD_TESTS=OFF > /dev/null
cmake --build "$BUILD_DIR" -j "$(nproc)" --target mcc > /dev/null
MCC=$BUILD_DIR/tools/mcc

mkdir -p "$OUT"
profile() {  # name, then mcc arguments
    local name=$1; shift
    local data
    data=$(mktemp /tmp/perf.XXXXXX.data)
    taskset -c 2 "$PERF" record -q -e cpu-clock -F 1999 -g -o "$data" -- "$MCC" "$@" > /dev/null
    "$PERF" script -i "$data" 2>/dev/null | "$FLAMEGRAPH_DIR/stackcollapse-perf.pl" > "/tmp/$name.folded"
    "$FLAMEGRAPH_DIR/flamegraph.pl" --title "$name" --width 1400 "/tmp/$name.folded" > "$OUT/$name.svg"
    {
        echo "# mcc $*"
        echo "# perf record -e cpu-clock -F 1999 -g"
        echo "# self time by shared object:"
        "$PERF" report -i "$data" --stdio --no-children --sort dso -g none --percent-limit 0.1 2>/dev/null \
            | grep -E '^ +[0-9.]+%' | sed -E 's/^ +/#   /; s/ +$//'
        echo "# self time by shared object and symbol:"
        "$PERF" report -i "$data" --stdio --no-children --sort dso,symbol -g none --percent-limit 0.5 2>/dev/null \
            | grep -E '^ +[0-9.]+%' | python3 "$ROOT/scripts/perf_summary.py" | head -25
    } > "$OUT/$name.txt"
    rm -f "$data"
    echo "$OUT/$name.svg"
}

G=bench/graphs
profile gelu_unfused "$G/gelu_chain.mcg" --dim M=2048 --passes none --bench --warmup 3 --reps 60
profile gelu_fused "$G/gelu_chain.mcg" --dim M=2048 --passes default --bench --warmup 3 --reps 150
profile mlp_block_fused "$G/mlp_block.mcg" --dim B=512 --passes default --bench --warmup 3 --reps 80

{
    echo "perf: $("$PERF" --version)"
    echo "kernel: $(uname -r)"
    echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
    echo "FlameGraph: $(git -C "$FLAMEGRAPH_DIR" rev-parse --short HEAD)"
    echo "minicompiler: $(git -C "$ROOT" rev-parse --short HEAD)$(git -C "$ROOT" diff --quiet || echo -dirty)"
} > "$OUT/environment.txt"
cat "$OUT/environment.txt"
