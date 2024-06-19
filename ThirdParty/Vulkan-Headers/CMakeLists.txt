cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    Vulkan-Headers
    GIT_REPOSITORY         https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG                e3c37e6e184a232e10b01dff5a065ce48c047f88
)

FetchContent_MakeAvailable(Vulkan-Headers)
