cmake_minimum_required(VERSION 3.20)
project(minicompiler VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(MINICOMPILER_BUILD_TESTS "Build the GoogleTest suites" ON)
option(MINICOMPILER_BUILD_BENCH "Build the benchmark programs" ON)
option(MINICOMPILER_BUILD_EXAMPLES "Build the example programs" ON)
option(MINICOMPILER_NATIVE_ARCH "Optimize for the build machine's CPU (-march=native)" ON)
option(MINICOMPILER_FRAME_POINTERS "Keep frame pointers so perf can walk call stacks" OFF)
option(MINICOMPILER_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

# Flags shared by every minicompiler target. They live on an interface target
# instead of add_compile_options() so third-party code is compiled untouched.
add_library(minicompiler_flags INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minicompiler_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Werror=return-type>)
    if(MINICOMPILER_NATIVE_ARCH)
        target_compile_options(minicompiler_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
    endif()
    if(MINICOMPILER_FRAME_POINTERS)
        target_compile_options(minicompiler_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer -mno-omit-leaf-frame-pointer>)
    endif()
    if(MINICOMPILER_SANITIZE)
        target_compile_options(minicompiler_flags INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer>)
        target_link_options(minicompiler_flags INTERFACE -fsanitize=address,undefined)
    endif()
elseif(MSVC)
    target_compile_options(minicompiler_flags INTERFACE $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->)
endif()

include(cmake/Dependencies.cmake)

enable_testing()

add_subdirectory(src)
add_subdirectory(tools)
if(MINICOMPILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
if(MINICOMPILER_BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(MINICOMPILER_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
