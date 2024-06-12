macro(FetchContent_DeclareShallowGit Name GIT_REPOSITORY GitRepository GIT_TAG GitTag)
    include(FetchContent)
    FetchContent_Declare(
        "${Name}"

        # This is what it'd look line if GIT_SHALLOW was indeed working:
        #GIT_REPOSITORY "${GitRepository}"
        #GIT_TAG        "${GitTag}"
        #GIT_SHALLOW    ON

        # Manual download mode instead:
        DOWNLOAD_COMMAND
            cd "${FETCHCONTENT_BASE_DIR}/${Name}-src" &&
            git init &&
            git fetch --depth=1 "${GitRepository}" "${GitTag}" &&
            git reset --hard FETCH_HEAD
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
set_directory_root_folder("SPIRV-Headers" "ThirdPartySPIRV-Headers")

add_subdirectory(SPIRV-Tools EXCLUDE_FROM_ALL)
set_directory_root_folder("SPIRV-Tools" "ThirdParty/SPIRV-Tools")

add_subdirectory(glslang EXCLUDE_FROM_ALL)
set_directory_root_folder("glslang" "ThirdParty/glslang")

add_subdirectory(abseil-cpp EXCLUDE_FROM_ALL)
set_directory_root_folder("abseil-cpp" "ThirdParty/abseil-cpp")

add_subdirectory(tint EXCLUDE_FROM_ALL)
set_directory_root_folder("tint" "ThirdParty/tint")
