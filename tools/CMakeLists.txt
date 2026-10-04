add_executable(mcc mcc.cpp)
target_include_directories(mcc PRIVATE ${PROJECT_SOURCE_DIR}/bench ${PROJECT_SOURCE_DIR}/src)
target_compile_definitions(mcc PRIVATE
    MINICOMPILER_BUILD_FLAGS="${CMAKE_BUILD_TYPE}, march_native=${MINICOMPILER_NATIVE_ARCH}")
target_link_libraries(mcc PRIVATE minicompiler minicompiler_flags)
