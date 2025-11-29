cmake_minimum_required (VERSION 3.19)

project(HLSLFunctionality)

add_executable(HLSLFunctionality main.cpp)

target_link_libraries(HLSLFunctionality glslang libtint SPIRV)
