cmake_minimum_required (VERSION 3.19)

project(MatrixOrder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(MatrixOrder main.cpp)

target_link_libraries(MatrixOrder glslang libtint SPIRV)
