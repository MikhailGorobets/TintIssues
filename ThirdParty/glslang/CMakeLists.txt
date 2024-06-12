cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    glslang
    GIT_REPOSITORY         https://github.com/KhronosGroup/glslang.git
    GIT_TAG                a92c61f8456fa9731c0b000a2c6fc52a740c2be7
)

FetchContent_MakeAvailable(glslang)
