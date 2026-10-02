include(GoogleTest)

add_executable(minicompiler_tests
    test_constant_folding.cpp
    test_cpu_backend.cpp
    test_dead_node_elimination.cpp
    test_dot_export.cpp
    test_graph.cpp
    test_memory_plan.cpp
    test_operator_fusion.cpp
    test_parser.cpp
    test_pipeline.cpp
    test_tensor.cpp
    test_util.cpp
)
target_include_directories(minicompiler_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(minicompiler_tests PRIVATE minicompiler minicompiler_flags GTest::gtest_main)

gtest_discover_tests(minicompiler_tests DISCOVERY_TIMEOUT 60)
