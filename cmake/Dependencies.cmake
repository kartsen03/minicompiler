# Third-party dependencies. Each one is taken from the system when available
# (apt: libeigen3-dev, libgtest-dev) and otherwise downloaded at configure
# time, pinned by version and SHA-256, so a plain checkout builds anywhere.

include(FetchContent)
# Downloaded archives get the extraction time as their timestamps, so a changed
# URL always rebuilds (CMake 3.24+ warns until a choice is made).
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

# --- Eigen (header-only) ----------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(TARGET Eigen3::Eigen)
    message(STATUS "minicompiler: using system Eigen ${Eigen3_VERSION}")
else()
    message(STATUS "minicompiler: Eigen3 not found, fetching 3.4.0")
    FetchContent_Declare(eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
        URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
        # Point SOURCE_SUBDIR at a directory with no CMakeLists.txt so the
        # archive is only unpacked; Eigen's own build (tests, docs) is skipped.
        SOURCE_SUBDIR do-not-build
    )
    FetchContent_MakeAvailable(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
endif()

# --- GoogleTest -------------------------------------------------------------
if(MINICOMPILER_BUILD_TESTS)
    find_package(GTest 1.10 QUIET)
    if(TARGET GTest::gtest_main)
        message(STATUS "minicompiler: using system GoogleTest ${GTest_VERSION}")
    else()
        message(STATUS "minicompiler: GoogleTest not found, fetching 1.15.2")
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
            URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
        )
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
endif()
