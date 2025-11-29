
# The GIT_SHALLOW option in FetchContent is buggy and does not actually perform
# a shallow clone. This macro fixes that. Also, we do not want to clone all
# submodules to reduce download time, therefore we add an option GIT_SUBMODULES
# to specify which submodules should be cloned (if any).
macro(FetchContent_DeclareShallowGit Name GIT_REPOSITORY GitRepository GIT_TAG GitTag GIT_SUBMODULES GitSubmodules)
    include(FetchContent)

    # This is what it'd look like if GIT_SHALLOW was indeed working:
    #GIT_REPOSITORY "${GitRepository}"
    #GIT_TAG        "${GitTag}"
    #GIT_SHALLOW     ON
    #GIT_SUBMODULES "${GitSubmodules}"

    # Manual download mode instead:
    FetchContent_Declare(
        ${Name}
        DOWNLOAD_COMMAND
            cd "${FETCHCONTENT_BASE_DIR}/${Name}-src" &&
            git init &&
            git remote add origin "${GitRepository}" &&
            git fetch --depth=1 origin "${GitTag}" &&
            git checkout FETCH_HEAD &&
            git submodule update --init --recursive --depth=1 ${GitSubmodules}
    )
endmacro()

function(find_targets_in_directory _RESULT _DIR)
    get_property(_subdirs DIRECTORY "${_DIR}" PROPERTY SUBDIRECTORIES)
    foreach(_subdir IN LISTS _subdirs)
        find_targets_in_directory(${_RESULT} "${_subdir}")
    endforeach()
    get_property(_SUB_TARGETS DIRECTORY "${_DIR}" PROPERTY BUILDSYSTEM_TARGETS)
    set(${_RESULT} ${${_RESULT}} ${_SUB_TARGETS} PARENT_SCOPE)
endfunction()

function(set_directory_root_folder _DIRECTORY _ROOT_FOLDER)
    find_targets_in_directory(_TARGETS ${_DIRECTORY})
    foreach(_TARGET IN LISTS _TARGETS)
        get_target_property(_FOLDER ${_TARGET} FOLDER)
        if(_FOLDER)
            set_target_properties(${_TARGET} PROPERTIES FOLDER "${_ROOT_FOLDER}/${_FOLDER}")
        else()
            set_target_properties(${_TARGET} PROPERTIES FOLDER "${_ROOT_FOLDER}")
        endif()
    endforeach()
endfunction()

add_subdirectory(SPIRV-Headers EXCLUDE_FROM_ALL)
set_directory_root_folder("SPIRV-Headers" "ThirdParty/SPIRV-Headers")

add_subdirectory(SPIRV-Tools EXCLUDE_FROM_ALL)
set_directory_root_folder("SPIRV-Tools" "ThirdParty/SPIRV-Tools")

add_subdirectory(glslang EXCLUDE_FROM_ALL)
set_directory_root_folder("glslang" "ThirdParty/glslang")

add_subdirectory(dawn EXCLUDE_FROM_ALL)
set_directory_root_folder("dawn" "ThirdParty/dawn")
