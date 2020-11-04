set(GFLAGS_BUILD_SHARED_LIBS ON)
set(GFLAGS_BUILD_STATIC_LIBS OFF)
set(GFLAGS_BUILD_gflags_LIB ON)
set(GFLAGS_BUILD_gflags_nothreads_LIB OFF)
set(GFLAGS_INSTALL_SHARED_LIBS ON)
set(GFLAGS_INSTALL_STATIC_LIBS OFF)
set(GFLAGS_INSTALL_HEADERS OFF)
set(GFLAGS_REGISTER_BUILD_DIR OFF)
set(GFLAGS_REGISTER_INSTALL_PREFIX OFF)
set(GFLAGS_LIBRARY_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})

# Add google test here, so that we do not propagate to it the WARNING_FLAGS
include(external/CMakeLists.txt.gflags.in)

# patch around gflags::gflags exporting their targets in two export sets
add_subdirectory(external/patches/cli-flags)

