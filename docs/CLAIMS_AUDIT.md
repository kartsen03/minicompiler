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
| 5 | The passes reduce computation overhead | **missing → verified, with limits** | `results/cpu/passes.json` (`bench/bench_cpu.cpp`): compute nodes 11 → 1 (GELU chain), 3 → 2 (matmul+bias+ReLU), 22 → 4 (MLP block); latency passes off vs on 2.33–3.20x faster for the GELU chain above 1 MB, 1.00–1.02x for matmul+bias+ReLU, 1.01–1.09x for the MLP block, and 0.91x (slower) for a 16 KB GELU; `docs/profiling/` explains why | Added the pass-ablation benchmark (no passes / DNE / DNE+fold / all, interleaved, median) and recorded it. The claim holds for node and kernel counts everywhere and for latency on elementwise-heavy graphs; matmul-dominated graphs gain little because GEMM is ~90% of their time |
| 6 | Benchmarked against PyTorch, in eager mode and with torch.compile | **missing → pending** | Harness: `bench/torch_compare.py` and `bench/mcgraph.py` (same graph files, same seeded weights and inputs, bit-identical RNG checked against C++ golden values). Results: not yet recorded | Wrote the harness: minicompiler, PyTorch eager op by op, PyTorch eager with its fused library ops (`F.gelu`, `addmm`, `F.batch_norm`), and `torch.compile`, single-threaded, with output cross-checks. **Blocked:** WSL needs `python3.12-venv` and `python3.12-dev` to install PyTorch and run `torch.compile` |
| 7 | The IR is visualized with Graphviz | **missing → verified** | `src/viz/dot_export.cpp`, `mcc --dump-dot`, `scripts/render_ir_diagrams.sh`; `docs/ir_diagrams/<graph>/` holds DOT and SVG for the input and after each of the 5 passes, for all 3 graphs; `tests/test_dot_export.cpp` renders the output through `dot` | Built DOT export (fused nodes list their kernel's instructions) and rendered the diagrams. The scaffold's `.gitignore` excluded `*.dot` and the diagram PNGs; that was changed so they are committed |
| 8 | Automated GoogleTest suites | **missing → verified** | `tests/` (73 test cases in 19 GoogleTest suites across 11 files, run through CTest); `.github/workflows/ci.yml` runs them on every push | Wrote the suites, including a random-graph generator that tracks interval bounds so it only builds numerically meaningful graphs |
| 9 | Built with CMake, version-controlled with Git, and tested on Linux | **partial → verified** | CMake: `CMakeLists.txt`, `cmake/Dependencies.cmake`. Git: incremental history. Linux: GitHub Actions on `ubuntu-latest` (Ubuntu 24.04) with GCC 13, Clang 18 and GCC Debug + ASan/UBSan, status badge in the README; local runs on Ubuntu 24.04 under WSL2 | The scaffold's CMake required an unused Google Benchmark package and built nothing; it now builds a library, the driver, tests and benchmarks, taking Eigen and GoogleTest from the system or a pinned download. There was no Linux test evidence; CI now provides it |
| 10 | Profiled hotspots with perf and flame graphs | **missing → verified** | `scripts/profile_cpu.sh`; `docs/profiling/` (flame graph SVGs, self-time summaries, environment, write-up). Found that copying every output cost 17.5% of a fused kernel's samples; removing it (`bd2306a`) cut the fused GELU from 12.6 to 10.7 ms (`docs/profiling/output_copy_fix_timing.json`) | Profiled with perf 6.8 on WSL2. WSL2 exposes no hardware PMU, so sampling uses the software `cpu-clock` event: time-based hotspots only, no cycle or cache-miss counts. Say "perf" in an interview, but be ready to explain this limit |

## Claims to remove or reword

None needs removing so far. Claim 6 is pending until the PyTorch comparison
runs. Two claims are true but should be stated precisely:

- **Claim 5.** "Reduce computation overhead" is true, but the size of the
  effect depends on the graph. A defensible phrasing:
  *fusion and folding cut compute nodes 22 → 4 on an MLP block and sped up
  elementwise-heavy graphs 2.3–3.2x on CPU.* Avoid implying a large
  end-to-end gain on matmul-heavy models.
- **Claim 10.** Profiling used `perf` with a software timer event because
  WSL2 has no hardware counters. Don't claim cache-miss or IPC analysis.
