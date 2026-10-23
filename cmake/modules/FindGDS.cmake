include(FindPackageHandleStandardArgs)

find_path(GDS_INCLUDE_DIR
  NAMES cufile.h
  HINTS ${GDS_ROOT} $ENV{GDS_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include targets/x86_64-linux/include
)

find_library(GDS_LIBRARY
  NAMES cufile
  HINTS ${GDS_ROOT} $ENV{GDS_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib
)

find_package_handle_standard_args(GDS
  REQUIRED_VARS GDS_INCLUDE_DIR GDS_LIBRARY
)

if(GDS_FOUND AND NOT TARGET NVIDIA::GDS)
  add_library(NVIDIA::GDS UNKNOWN IMPORTED)
  set_target_properties(NVIDIA::GDS PROPERTIES
    IMPORTED_LOCATION "${GDS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${GDS_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(GDS_INCLUDE_DIR GDS_LIBRARY)
