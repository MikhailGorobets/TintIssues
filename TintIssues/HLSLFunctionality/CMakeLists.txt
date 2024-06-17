cmake_minimum_required (VERSION 3.19)

project(HLSLFunctionality)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(HLSLFunctionality main.cpp)

target_link_libraries(HLSLFunctionality glslang libtint SPIRV)
