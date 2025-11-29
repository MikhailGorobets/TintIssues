cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    SPIRV-Tools
    GIT_REPOSITORY         https://github.com/KhronosGroup/SPIRV-Tools.git
    GIT_TAG                7f2d9ee926f98fc77a3ed1e1e0f113b8c9c49458
    GIT_SUBMODULES         ""
)

FetchContent_MakeAvailable(SPIRV-Tools)
