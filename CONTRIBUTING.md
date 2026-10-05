# Contributing to minicompiler

The conventions the code follows. Build and test instructions are in the
[README](README.md#quick-start).

## Language & Build
- Language: C++17
- Build system: CMake 3.20+

## Dependencies
Taken from the system when installed (apt), otherwise fetched by CMake at a
pinned version and hash (see `cmake/Dependencies.cmake`):
- Eigen
- GoogleTest

Tools, not build dependencies:
- Graphviz (`dot`) renders the IR diagrams
- `perf` and FlameGraph for CPU profiling
- Optional: CUDA toolkit for the CUDA backend

Benchmarks use a small in-repo harness (`bench/harness.hpp`) rather than
Google Benchmark, so warmup, repetition count and the reported statistic
(median) are explicit and identical for CPU and CUDA-event timing.

## Style
- `snake_case` for functions and variables
- `PascalCase` for classes, structs, and enums
- Header guards: use `#pragma once`

## Layout
- Headers: `include/minicompiler/`
- Sources: `src/`
- Tests: `tests/`
- Benchmarks: `bench/`
- Examples: `examples/`

## API Boundaries
- Public API headers go in `include/minicompiler/`
- Internal-only headers stay in `src/`
- All public classes use the namespace `minicompiler`

## Error Handling
- No exceptions for control flow
- Use `std::optional` or `std::expected`-like return types for fallible operations

## Benchmarks and Results
- Every number in the README is generated from `results/*.json` by
  `scripts/results_tables.py`; record results with `scripts/run_cpu_benchmarks.sh`
  or `scripts/run_gpu_benchmarks.sh`, never by hand
- Each result file records the commit (with `-dirty` when code was uncommitted),
  the machine and the software versions
