add_library(minicompiler STATIC
    fused_program.cpp
    graph.cpp
    graph_builder.cpp
    ops.cpp
    parser.cpp
    random.cpp
    viz/dot_export.cpp
)

target_include_directories(minicompiler PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(minicompiler
    PUBLIC Eigen3::Eigen
    PRIVATE minicompiler_flags
)
