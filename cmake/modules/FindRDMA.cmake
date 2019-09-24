# FindNUMA.cmake
#
# Finds librdmacm and libibverbs
#
# This will define the following variables
#    RDMA_FOUND
#    RDMA_LIBRARIES
#    RDMA_INCLUDE_DIRS
#
# and the following imported targets
#
#     RDMA::RDMA

find_package(PkgConfig)
pkg_check_modules(PC_RDMA QUIET RDMA)

find_path(RDMA_RDMACM_INCLUDE_DIRS
    NAMES rdma/rdma_cma.h
    HINTS ${RDMA_ROOT_DIR}/include
    )

find_path(RDMA_IBVERBS_INCLUDE_DIRS
    NAMES infiniband/verbs.h
    HINTS ${RDMA_ROOT_DIR}/include
    )

find_library(RDMA_RDMACM_LIBRARIES
    NAMES rdmacm
    HINTS ${RDMA_ROOT_DIR}/lib
    )
find_library(RDMA_IBVERBS_LIBRARIES
    NAMES ibverbs
    HINTS ${RDMA_ROOT_DIR}/lib
    )

set(RDMA_INCLUDE_DIRS ${RDMA_RDMACM_INCLUDE_DIRS} ${RDMA_RDMACM_INCLUDE_DIRS})
set(RDMA_LIBRARIES ${RDMA_RDMACM_LIBRARIES} ${RDMA_IBVERBS_LIBRARIES})

set(RDMA_VERSION ${PC_RDMA_VERSION})

mark_as_advanced(RDMA_FOUND RDMA_INCLUDE_DIRS RDMA_LIBRARIES RDMA_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RDMA
    DEFAULT_MSG RDMA_INCLUDE_DIRS RDMA_LIBRARIES
    )

if(RDMA_FOUND AND NOT TARGET RDMA::RDMA)
    add_library(RDMA::RDMA UNKNOWN IMPORTED)
    set_target_properties(RDMA::RDMA
        PROPERTIES
            IMPORTED_LOCATION ${RDMA_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${RDMA_INCLUDE_DIRS}
        )
endif()
