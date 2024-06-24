cmake_minimum_required (VERSION 3.19)

project(LinkDxguidD3D11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(LinkDxguidD3D11 main.cpp)

target_link_libraries(LinkDxguidD3D11 webgpu_dawn)
