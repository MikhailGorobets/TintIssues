cmake_minimum_required (VERSION 3.19)

project(StaticArrayOfResources)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(StaticArrayOfResources main.cpp)

target_link_libraries(StaticArrayOfResources glslang libtint SPIRV)
