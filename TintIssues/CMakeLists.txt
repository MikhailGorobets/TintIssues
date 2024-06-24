cmake_minimum_required (VERSION 3.19)

add_subdirectory(ComparisonIntTypes)
add_subdirectory(HLSLFunctionality)
add_subdirectory(MatrixOrder)
add_subdirectory(NoMatchingConstructor)

set_directory_root_folder("ComparisonIntTypes" "TintIssues")
set_directory_root_folder("HLSLFunctionality" "TintIssues")
set_directory_root_folder("MatrixOrder" "TintIssues")
set_directory_root_folder("NoMatchingConstructor" "TintIssues")
