cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    SPIRV-Tools
    GIT_REPOSITORY         https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG                65d30c31508e47d34f7698f511aea4800e3d44fe
)

FetchContent_MakeAvailable(SPIRV-Tools)
