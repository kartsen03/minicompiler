# minicompiler

## Language & Build
- Language: C++17
- Build system: CMake 3.20+

## Dependencies
System-installed via apt:
- Eigen
- GoogleTest
- Google Benchmark
- Graphviz

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
