cmake_minimum_required (VERSION 3.19)

project(ComparisonIntTypes)

add_executable(ComparisonIntTypes main.cpp)

target_link_libraries(ComparisonIntTypes glslang libtint SPIRV)
