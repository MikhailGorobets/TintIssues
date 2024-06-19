cmake_minimum_required (VERSION 3.19)

project(NoMatchingConstructor)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(NoMatchingConstructor main.cpp)

target_link_libraries(NoMatchingConstructor glslang libtint SPIRV)
