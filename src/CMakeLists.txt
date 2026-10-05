add_library(minicompiler STATIC
    cpu/cpu_backend.cpp
    cpu/kernels.cpp
    cuda/codegen.cpp
    fused_program.cpp
    graph.cpp
    graph_builder.cpp
    ops.cpp
    parser.cpp
    passes/constant_folding.cpp
    passes/dead_node_elimination.cpp
    passes/operator_fusion.cpp
    passes/pipeline.cpp
    random.cpp
    runtime/backend.cpp
    runtime/host_tensor.cpp
    runtime/memory_plan.cpp
    viz/dot_export.cpp
)

target_include_directories(minicompiler
    PUBLIC ${PROJECT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(minicompiler
    PUBLIC Eigen3::Eigen
    PRIVATE minicompiler_flags
)

# The CUDA backend. cuda/codegen.cpp (above) is plain C++ and always built, so
# the kernel generator is tested even without a GPU.
if(MINICOMPILER_HAVE_CUDA)
    target_sources(minicompiler PRIVATE
        cuda/cuda_backend.cpp
        cuda/device_info.cpp
        cuda/kernel_compiler.cpp
        cuda/matmul_kernels.cu
        cuda/matmul_tensor_core.cu
    )
    target_compile_definitions(minicompiler PUBLIC MINICOMPILER_HAVE_CUDA=1)
    target_compile_options(minicompiler PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_link_libraries(minicompiler
        PUBLIC CUDA::cudart
        PRIVATE CUDA::nvrtc CUDA::cuda_driver CUDA::nvml
    )
endif()
