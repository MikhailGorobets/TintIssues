cmake_minimum_required (VERSION 3.19)

project(ComparisonIntTypes)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(ComparisonIntTypes ComparisonIntTypes/main.cpp)

target_link_libraries(ComparisonIntTypes glslang libtint SPIRV)
