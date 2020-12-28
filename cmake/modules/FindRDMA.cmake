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

find_path(RDMA_MLX5_INCLUDE_DIRS
    NAMES infiniband/mlx5dv.h
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
find_library(RDMA_MLX5_LIBRARIES
    NAMES mlx5
    HINTS ${RDMA_ROOT_DIR}/lib
    )

set(RDMA_INCLUDE_DIRS ${RDMA_RDMACM_INCLUDE_DIRS} ${RDMA_IBVERBS_INCLUDE_DIRS} ${RDMA_MLX5_INCLUDE_DIRS})
set(RDMA_LIBRARIES ${RDMA_RDMACM_LIBRARIES} ${RDMA_IBVERBS_LIBRARIES} ${RDMA_MLX5_LIBRARIES})

set(RDMA_VERSION ${PC_RDMA_VERSION})

mark_as_advanced(RDMA_FOUND RDMA_INCLUDE_DIRS RDMA_LIBRARIES RDMA_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RDMA
    DEFAULT_MSG RDMA_INCLUDE_DIRS RDMA_LIBRARIES
    )

if(RDMA_FOUND AND NOT TARGET RDMA::RDMA)
    add_library(RDMA::MLX5DV UNKNOWN IMPORTED)
    set_target_properties(RDMA::MLX5DV
        PROPERTIES
            IMPORTED_LOCATION ${RDMA_MLX5_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${RDMA_MLX5_INCLUDE_DIRS}
        )
    add_library(RDMA::IBVERBS UNKNOWN IMPORTED)
    set_target_properties(RDMA::IBVERBS
        PROPERTIES
            IMPORTED_LOCATION ${RDMA_IBVERBS_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${RDMA_IBVERBS_INCLUDE_DIRS}
        )
    add_library(RDMA::RDMACM UNKNOWN IMPORTED)
    set_target_properties(RDMA::RDMACM
        PROPERTIES
            IMPORTED_LOCATION ${RDMA_RDMACM_LIBRARIES}
            INTERFACE_INCLUDE_DIRECTORIES ${RDMA_RDMACM_INCLUDE_DIRS}
        )
    add_library(RDMA::RDMA INTERFACE IMPORTED)
    set_property(TARGET RDMA::RDMA PROPERTY
            INTERFACE_LINK_LIBRARIES RDMA::RDMACM RDMA::IBVERBS RDMA::MLX5DV)
endif()
