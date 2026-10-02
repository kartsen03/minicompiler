add_executable(bench_cpu bench_cpu.cpp)
target_compile_definitions(bench_cpu PRIVATE
    MINICOMPILER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    MINICOMPILER_BUILD_FLAGS="${CMAKE_BUILD_TYPE}, march_native=${MINICOMPILER_NATIVE_ARCH}")
target_link_libraries(bench_cpu PRIVATE minicompiler minicompiler_flags)
