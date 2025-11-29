cmake_minimum_required (VERSION 3.19)

add_subdirectory(HLSLFunctionality)
set_directory_root_folder("HLSLFunctionality" "TintIssues")

# We disable these build because tint change API
if (FALSE)
    add_subdirectory(BindingVariableParser)
    add_subdirectory(ComparisonIntTypes)
    add_subdirectory(MatrixOrder)
    add_subdirectory(NoMatchingConstructor)
    add_subdirectory(StaticArrayOfResources)

    set_directory_root_folder("BindingVariableParser" "TintIssues")
    set_directory_root_folder("ComparisonIntTypes" "TintIssues")
    set_directory_root_folder("MatrixOrder" "TintIssues")
    set_directory_root_folder("NoMatchingConstructor" "TintIssues")
    set_directory_root_folder("StaticArrayOfResources" "TintIssues")
endif()
