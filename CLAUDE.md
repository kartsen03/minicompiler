# minicompiler

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
