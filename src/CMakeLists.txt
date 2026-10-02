add_library(minicompiler STATIC
    cpu/cpu_backend.cpp
    cpu/kernels.cpp
    fused_program.cpp
    graph.cpp
    graph_builder.cpp
    ops.cpp
    parser.cpp
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
