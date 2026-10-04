include(GoogleTest)

add_executable(minicompiler_tests
    test_constant_folding.cpp
    test_cpu_backend.cpp
    test_cuda_codegen.cpp
    test_dead_node_elimination.cpp
    test_dot_export.cpp
    test_end_to_end.cpp
    test_graph.cpp
    test_memory_plan.cpp
    test_operator_fusion.cpp
    test_parser.cpp
    test_pipeline.cpp
    test_tensor.cpp
    test_util.cpp
)
if(MINICOMPILER_HAVE_CUDA)
    target_sources(minicompiler_tests PRIVATE test_cuda_backend.cpp)
endif()
target_include_directories(minicompiler_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(minicompiler_tests PRIVATE MINICOMPILER_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
target_link_libraries(minicompiler_tests PRIVATE minicompiler minicompiler_flags GTest::gtest_main)

gtest_discover_tests(minicompiler_tests DISCOVERY_TIMEOUT 60)
