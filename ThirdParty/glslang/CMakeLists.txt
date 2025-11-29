cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    glslang
    GIT_REPOSITORY         https://github.com/KhronosGroup/glslang.git
    GIT_TAG                a57276bf558f5cf94d3a9854ebdf5a2236849a5a
    GIT_SUBMODULES         ""
)

FetchContent_MakeAvailable(glslang)
