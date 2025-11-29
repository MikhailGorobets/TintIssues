cmake_minimum_required (VERSION 3.19)

project(StaticArrayOfResources)

add_executable(StaticArrayOfResources main.cpp)

target_link_libraries(StaticArrayOfResources glslang libtint SPIRV)
