cmake_minimum_required(VERSION 3.20)
project(minicompiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(examples)
