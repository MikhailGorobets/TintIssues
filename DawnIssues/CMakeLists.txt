cmake_minimum_required (VERSION 3.19)

# We disable these build because dawn change API
if (FALSE)
    add_subdirectory(LinkDxguidD3D11)
    set_directory_root_folder("LinkDxguidD3D11" "DawnIssues")
endif()
