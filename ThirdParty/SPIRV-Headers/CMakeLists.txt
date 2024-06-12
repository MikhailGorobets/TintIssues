cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    SPIRV-Headers
    GIT_REPOSITORY         https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG                eb49bb7b1136298b77945c52b4bbbc433f7885de
)

FetchContent_MakeAvailable(SPIRV-Headers)
