cmake_minimum_required (VERSION 3.19)

project(BindingVariableParser)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(BindingVariableParser main.cpp)

target_link_libraries(BindingVariableParser glslang SPIRV libtint)

target_include_directories(BindingVariableParser PRIVATE
        "${spirv-tools_SOURCE_DIR}"
        "${spirv-tools_SOURCE_DIR}/include"
        "${spirv-tools_SOURCE_DIR}/source"
        "${spirv-tools_BINARY_DIR}"
)
