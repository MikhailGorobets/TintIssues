cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    SPIRV-Headers
    GIT_REPOSITORY         https://github.com/KhronosGroup/SPIRV-Headers.git
    GIT_TAG                01e0577914a75a2569c846778c2f93aa8e6feddd
    GIT_SUBMODULES         ""
)

FetchContent_MakeAvailable(SPIRV-Headers)
