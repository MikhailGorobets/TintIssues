cmake_minimum_required (VERSION 3.19)

project(NoMatchingConstructor)

add_executable(NoMatchingConstructor main.cpp)

target_link_libraries(NoMatchingConstructor glslang libtint SPIRV)
