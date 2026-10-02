include(GoogleTest)

add_executable(minicompiler_tests
    test_graph.cpp
    test_tensor.cpp
)
target_link_libraries(minicompiler_tests PRIVATE minicompiler minicompiler_flags GTest::gtest_main)

gtest_discover_tests(minicompiler_tests DISCOVERY_TIMEOUT 60)
