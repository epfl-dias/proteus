# FindNUMA.cmake
#
# Finds libnuma
#
# This will define the following variables
#    NUMA_FOUND
#    NUMA_INCLUDE_DIRS
#
# and the following imported targets
#
#     NUMA::NUMA

find_package(PkgConfig)
pkg_check_modules(PC_NUMA QUIET NUMA)

find_path(NUMA_INCLUDE_DIRS
    NAMES numa.h numaif.h
    HINTS ${NUMA_ROOT_DIR}/include
    )

find_library(NUMA_LIBRARIES
    NAMES numa
    HINTS ${NUMA_ROOT_DIR}/lib
    )

set(NUMA_VERSION ${PC_NUMA_VERSION})

mark_as_advanced(NUMA_FOUND NUMA_INCLUDE_DIRS NUMA_LIBRARIES NUMA_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA
    DEFAULT_MSG NUMA_INCLUDE_DIRS NUMA_LIBRARIES
    )

if(NUMA_FOUND AND NOT TARGET NUMA::NUMA)
    add_library(NUMA::NUMA UNKNOWN IMPORTED)
    set_target_properties(NUMA::NUMA
        PROPERTIES
            IMPORTED_LOCATION ${NUMA_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${NUMA_INCLUDE_DIRS}
        )
endif()
