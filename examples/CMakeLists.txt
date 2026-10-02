add_executable(quickstart quickstart.cpp)
target_link_libraries(quickstart PRIVATE minicompiler minicompiler_flags)

# Running the example is part of the test suite, so it cannot rot.
add_test(NAME example.quickstart COMMAND quickstart)
