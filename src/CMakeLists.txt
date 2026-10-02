add_library(minicompiler INTERFACE)

target_include_directories(minicompiler INTERFACE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(minicompiler INTERFACE Eigen3::Eigen)
