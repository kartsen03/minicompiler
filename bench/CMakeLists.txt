add_executable(bench_cpu bench_cpu.cpp)
target_compile_definitions(bench_cpu PRIVATE
    MINICOMPILER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
    MINICOMPILER_BUILD_FLAGS="${CMAKE_BUILD_TYPE}, march_native=${MINICOMPILER_NATIVE_ARCH}")
target_link_libraries(bench_cpu PRIVATE minicompiler minicompiler_flags)

# C interface for timing minicompiler from Python in the same process as
# PyTorch (bench/torch_compare.py loads it with ctypes).
add_library(minicompiler_capi SHARED capi.cpp)
target_link_libraries(minicompiler_capi PRIVATE minicompiler minicompiler_flags)

if(MINICOMPILER_HAVE_CUDA)
    add_executable(bench_cuda bench_cuda.cpp)
    target_compile_definitions(bench_cuda PRIVATE
        MINICOMPILER_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
        MINICOMPILER_BUILD_FLAGS="${CMAKE_BUILD_TYPE}, march_native=${MINICOMPILER_NATIVE_ARCH}")
    target_include_directories(bench_cuda PRIVATE ${PROJECT_SOURCE_DIR}/src)
    # cuBLAS is the reference baseline in the benchmark only, never in the library.
    target_link_libraries(bench_cuda PRIVATE minicompiler minicompiler_flags CUDA::cublas)
endif()
