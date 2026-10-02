include(GoogleTest)

add_executable(minicompiler_tests
    test_dot_export.cpp
    test_graph.cpp
    test_memory_plan.cpp
    test_parser.cpp
    test_tensor.cpp
    test_util.cpp
)
target_include_directories(minicompiler_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(minicompiler_tests PRIVATE minicompiler minicompiler_flags GTest::gtest_main)

gtest_discover_tests(minicompiler_tests DISCOVERY_TIMEOUT 60)
