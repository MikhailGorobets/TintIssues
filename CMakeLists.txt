cmake_minimum_required (VERSION 3.19)

project(TintIssuesProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(ThirdParty)
add_subdirectory(TintIssues)
add_subdirectory(DawnIssues)
