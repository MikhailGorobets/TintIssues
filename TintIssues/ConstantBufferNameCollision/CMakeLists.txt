cmake_minimum_required (VERSION 3.19)

project(ConstantBufferNameCollision)

add_executable(ConstantBufferNameCollision main.cpp)

target_link_libraries(ConstantBufferNameCollision glslang libtint SPIRV)
