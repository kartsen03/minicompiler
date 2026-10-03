add_executable(bench_cpu bench_cpu.cpp)
target_compile_definitions(bench_cpu PRIVATE
    MINICOMPILER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    MINICOMPILER_BUILD_FLAGS="${CMAKE_BUILD_TYPE}, march_native=${MINICOMPILER_NATIVE_ARCH}")
target_link_libraries(bench_cpu PRIVATE minicompiler minicompiler_flags)

# C interface for timing minicompiler from Python in the same process as
# PyTorch (bench/torch_compare.py loads it with ctypes).
add_library(minicompiler_capi SHARED capi.cpp)
target_link_libraries(minicompiler_capi PRIVATE minicompiler minicompiler_flags)
