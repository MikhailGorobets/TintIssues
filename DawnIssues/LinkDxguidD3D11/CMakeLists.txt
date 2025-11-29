cmake_minimum_required (VERSION 3.19)

project(LinkDxguidD3D11)

add_executable(LinkDxguidD3D11 main.cpp)

target_link_libraries(LinkDxguidD3D11 webgpu_dawn)
