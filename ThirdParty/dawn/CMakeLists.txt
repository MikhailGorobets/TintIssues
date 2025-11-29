cmake_minimum_required (VERSION 3.11)

set(DAWN_SUBMODULES
    third_party/abseil-cpp
    third_party/jinja2
    third_party/markupsafe
    third_party/libprotobuf-mutator
    third_party/protobuf
    third_party/webgpu-headers
)

FetchContent_DeclareShallowGit(
    dawn
    GIT_REPOSITORY  https://dawn.googlesource.com/dawn
    GIT_TAG         042e54a9f93e6d2fed503aabe68003dff1705805
    GIT_SUBMODULES  "${DAWN_SUBMODULES}"
)

set(TINT_BUILD_AS_OTHER_OS ON CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_READER  ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_IR_BINARY   ON CACHE BOOL "" FORCE)
set(TINT_BUILD_WGSL_READER ON CACHE BOOL "" FORCE)
set(TINT_BUILD_HLSL_WRITER ON CACHE BOOL "" FORCE)

set(TINT_BUILD_GLSL_WRITER     OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_MSL_WRITER      OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_SPV_WRITER      OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_GLSL_VALIDATOR  OFF CACHE BOOL "" FORCE)
set(TINT_ENABLE_IR_VALIDATION  OFF CACHE BOOL "" FORCE)

set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_DOCS      OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_SAMPLES     OFF CACHE BOOL "" FORCE)

set(DAWN_ENABLE_VULKAN     OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_D3D11      ON  CACHE BOOL "" FORCE)
set(DAWN_ENABLE_D3D12      OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_OPENGLES   OFF CACHE BOOL "" FORCE)

set(DAWN_USE_GLFW                 OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_NULL              OFF CACHE BOOL "" FORCE)
set(DAWN_USE_WINDOWS_UI           OFF CACHE BOOL "" FORCE)
set(DAWN_ENABLE_SPIRV_VALIDATION  OFF CACHE BOOL "" FORCE)

set(DAWN_SPIRV_TOOLS_DIR   "${spirv-tools_SOURCE_DIR}"   CACHE STRING "Directory in which to find SPIRV-Tools"   FORCE)
set(DAWN_SPIRV_HEADERS_DIR "${SPIRV-Headers_SOURCE_DIR}" CACHE STRING "Directory in which to find SPIRV-Headers" FORCE)

set(BUILD_SAMPLES      OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS  OFF CACHE BOOL "")

FetchContent_MakeAvailable(dawn)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(tint_api PRIVATE -Wno-switch-default)
endif()

FetchContent_GetProperties(dawn SOURCE_DIR dawn_SOURCE_DIR)
FetchContent_GetProperties(tint SOURCE_DIR tint_SOURCE_DIR)
