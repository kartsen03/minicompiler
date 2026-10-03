# Resume claims audit

Each claim made about minicompiler, checked against the repository.

**Status** has two parts. The first is what the repository supported when the
audit started, at commit `c3523d8` (two commits: a CMake scaffold and a
`Tensor` shape/dtype header, with no IR, passes, backend, tests or
benchmarks). The second is the status after the work recorded below.
**Evidence** points at what a reviewer can open or run.

| # | Claim | Status | Evidence | What changed |
|---|---|---|---|---|
| 1 | A modern C++ compiler for ML compute graphs | **missing → verified** | `tools/mcc.cpp` (driver), `src/parser.cpp` (graph programs in the `.mcg` format), IR, passes and backend under `include/` and `src/`; C++17 with value-semantic IR, RAII, `std::shared_ptr` payloads and `Result<T>` errors instead of exceptions | Built the compiler: text frontend, IR, pass pipeline, memory planner, CPU backend and command-line driver |
| 2 | A DAG-based intermediate representation | **missing → verified** | `include/minicompiler/graph.hpp`; `Graph::verify()` checks that every operand precedes its user (topological order, hence acyclic); `tests/test_graph.cpp` | Built an SSA-style DAG IR with shape inference (NumPy broadcasting, matmul), a builder with sticky errors, def-use queries and node statistics |
| 3 | Three optimization passes: dead-node elimination, constant folding and operator fusion | **missing → verified** | `src/passes/`; `tests/test_dead_node_elimination.cpp` (removes unreachable nodes, keeps inputs, idempotent), `tests/test_constant_folding.cpp` (folds `sqrt(2/pi)`, BatchNorm's `gamma/sqrt(var+eps)` and matmuls of constants to exact values), `tests/test_operator_fusion.cpp` (exact groups for 8 graph shapes, including the would-be cycle through a matmul); `tests/test_end_to_end.cpp` checks optimized and unoptimized graphs agree on random inputs within `1e-5·\|ref\| + 1e-5·max\|ref\|`, on the three benchmark graphs and 300 random graphs | Implemented the three passes and a pipeline (`dne, fold, dne, fuse, dne`) that verifies the graph after every pass |
| 4 | An Eigen CPU backend that executes the optimized graph | **missing → verified** | `src/cpu/kernels.cpp` (Eigen elementwise and broadcasting kernels, Eigen GEMM, block-wise fused kernels), `src/cpu/cpu_backend.cpp`, `src/runtime/memory_plan.cpp`; `tests/test_cpu_backend.cpp` compares every op to a double-precision reference (bit-exact for `+ - * /`, neg, relu; 2 ulp for sqrt; 8 ulp for transcendental functions; the γₖ error bound for matmul) | Built the backend, selected at runtime with `create_backend("cpu")`, with lifetime-based buffer reuse |
| 5 | The passes reduce computation overhead | **missing → verified, with limits** | `results/cpu/passes.json` (`bench/bench_cpu.cpp`, 3 runs, interleaved, paired ratios): compute nodes 11 → 1 (GELU chain), 3 → 2 (matmul+bias+ReLU), 22 → 4 (MLP block). Latency, passes off vs on: GELU chain 2.34–2.91x faster for 1 MB and up (1.06x at 16 KB), matmul+bias+ReLU 1.00–1.01x, MLP block 1.01–1.11x. Constant folding alone: 0.99–1.03x. `docs/profiling/` explains why | Added the pass-ablation benchmark (no passes / DNE / DNE+fold / all) and recorded it. The claim holds for node and kernel counts everywhere and for latency on elementwise-heavy graphs; matmul-dominated graphs gain little because GEMM is ~90% of their time, and folding only removes scalar arithmetic |
| 6 | Benchmarked against PyTorch, in eager mode and with torch.compile | **missing → verified** | `results/cpu/torch_compare.json` and `torch_compare_tuned_malloc.json` (raw runs in `results/cpu/runs/`), `results/cpu/tanh.json`; harness `bench/torch_compare.py`, `bench/mcgraph.py`, `bench/capi.cpp`. Same graph files, shapes, seeded weights and inputs; outputs agree to 5.4e-7. Results: GELU chain 2.73–3.87x faster than PyTorch's fused `F.gelu` and 3.01–4.05x faster than `torch.compile` (1 MB and up), largely because Eigen's `tanh` is 3.9x cheaper and less accurate (4.75 vs 0.51 ulp); PyTorch wins matmul+bias+ReLU by 1–15%; the MLP block is within 0.86–1.20x | Wrote the harness: minicompiler (in-process through a C interface), PyTorch eager op by op, PyTorch with its fused library ops, and `torch.compile`, single-threaded, timed interleaved in one process, 3 runs, plus a run with glibc's allocator tuned so PyTorch avoids per-call page faults |
| 7 | The IR is visualized with Graphviz | **missing → verified** | `src/viz/dot_export.cpp`, `mcc --dump-dot`, `scripts/render_ir_diagrams.sh`; `docs/ir_diagrams/<graph>/` holds DOT and SVG for the input and after each of the 5 passes, for all 3 graphs; `tests/test_dot_export.cpp` renders the output through `dot` | Built DOT export (fused nodes list their kernel's instructions) and rendered the diagrams. The scaffold's `.gitignore` excluded `*.dot` and the diagram PNGs; that was changed so they are committed |
| 8 | Automated GoogleTest suites | **missing → verified** | `tests/` (80 test cases in 19 GoogleTest suites across 11 files, plus the example program, run through CTest); `.github/workflows/ci.yml` runs them on every push | Wrote the suites, including a random-graph generator that tracks interval bounds so it only builds numerically meaningful graphs |
| 9 | Built with CMake, version-controlled with Git, and tested on Linux | **partial → verified** | CMake: `CMakeLists.txt`, `cmake/Dependencies.cmake`. Git: incremental history. Linux: GitHub Actions on `ubuntu-latest` (Ubuntu 24.04) with GCC 13, Clang 18 and GCC Debug + ASan/UBSan, status badge in the README; local runs on Ubuntu 24.04 under WSL2 | The scaffold's CMake required an unused Google Benchmark package and built nothing; it now builds a library, the driver, tests and benchmarks, taking Eigen and GoogleTest from the system or a pinned download. There was no Linux test evidence; CI now provides it |
| 10 | Profiled hotspots with perf and flame graphs | **missing → verified** | `scripts/profile_cpu.sh`; `docs/profiling/` (flame graph SVGs, self-time summaries, environment, write-up). Found that copying every output cost 17.5% of a fused kernel's samples; removing it (`bd2306a`) cut the fused GELU from 12.6 to 10.7 ms (`docs/profiling/output_copy_fix_timing.json`) | Profiled with perf 6.8 on WSL2. WSL2 exposes no hardware PMU, so sampling uses the software `cpu-clock` event: time-based hotspots only, no cycle or cache-miss counts. Say "perf" in an interview, but be ready to explain this limit |

## Claims to remove or reword

None needs removing. Three are true but need careful wording:

- **Claim 5.** "Reduce computation overhead" is true, but the effect depends
  on the graph. A defensible phrasing: *fusion and folding cut compute nodes
  22 → 4 on an MLP block and made elementwise-heavy graphs 2.3–2.9x faster on
  CPU.* Avoid implying a large end-to-end gain on matmul-heavy models
  (1.00–1.11x measured).
- **Claim 6.** "Benchmarked against PyTorch" is true. "Faster than PyTorch"
  needs qualifying: the GELU win over PyTorch's fused kernels comes mostly from
  Eigen's cheaper, less accurate `tanh`, and PyTorch wins on matmul-heavy
  graphs. Say what was compared and where each side wins.
- **Claim 10.** Profiling used `perf` with a software timer event because
  WSL2 has no hardware counters. Don't claim cache-miss or IPC analysis.

## How the numbers were measured

The CPU numbers come from an i7-12700H laptop under WSL2 with the Windows
"Silent" power plan. A pinned core's speed there drifts by up to 2x within a
minute, so every comparison is timed in interleaved rounds within one process,
reported as the median of per-round ratios, and repeated in three runs; the
result files keep every raw run. An earlier PyTorch comparison that timed
minicompiler in separate processes was discarded because of this drift.
