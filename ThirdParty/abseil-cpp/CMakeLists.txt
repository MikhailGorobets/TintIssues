cmake_minimum_required (VERSION 3.11)

FetchContent_DeclareShallowGit(
    abseil-cpp
    GIT_REPOSITORY         https://github.com/abseil/abseil-cpp.git
    GIT_TAG                54e1f14c6f9b1a764ffdf8c1aea7e823f95f3d01
)

set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(abseil-cpp)

# Hack for dawn
#add_library(libabsl ALIAS absl_base)
