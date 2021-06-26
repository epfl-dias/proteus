project(platform VERSION 0.1 LANGUAGES CXX)

find_package(NUMA REQUIRED)
find_package(RDMA REQUIRED)

# Pure, regular C++ files
set(platform_cxx
  lib/common/common.cpp
  lib/common/gpu/gpu-common.cpp
  lib/memory/block-manager.cpp
  lib/memory/memory-allocator.cpp
  lib/memory/memory-manager.cpp
  lib/network/infiniband/devices/ib.cpp
  lib/network/infiniband/private/ib_impl.cpp
  lib/network/infiniband/infiniband-client.cpp
  lib/network/infiniband/infiniband-handler.cpp
  lib/network/infiniband/infiniband-manager.cpp
  lib/network/infiniband/infiniband-server.cpp
  lib/storage/mmap-file.cpp
  lib/storage/storage-manager.cpp
  lib/topology/affinity_manager.cpp
  lib/topology/device-manager.cpp
  lib/topology/topology.cpp
  lib/util/percentile.cpp
  lib/util/profiling.cpp
  lib/util/stacktrace.cpp
  lib/util/time-registry.cpp
  lib/util/radix/aggregations/radix-aggr.cpp
  lib/util/radix/joins/radix-join.cpp
  )

# C++ files requiring CUDA headers
set(platform_cxx_cu
  #operators/gpu/gpu-partitioned-hash-join-chained-helpers.cpp
  )

# Pure CUDA files
set(platform_cu
  # LSC: 30.11.2018
  #	Work around issue in:
  #	  /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/detail/cub/util_debug.cuh 117-136
  #	Which fails with va_printf undefined.
  #
  #	For now we build two times this, as otherwise we get linking and/or
  #	runtime error as LLVM cannot yet generate relocatable ptx code.
  lib/memory/buffer-manager.cu
  lib/util/sort/gpu/sort-util.cu
  )

foreach (target ${platform_cu})
  set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach ()

foreach (target ${platform_cxx_cu} ${platform_cu})
  set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach ()

#-----------------------------------------------------------------------------
# LSC: Hopefully we will soon be able to remove this section, where we call
#      straight to NVCC
set(CUDA_WORKDIR ${CMAKE_CURRENT_BINARY_DIR}/cubin)
file(MAKE_DIRECTORY ${CUDA_WORKDIR})

if (CUDA_FOUND)
  set(platform_cu_objs
    "${CUDA_WORKDIR}/buffer-manager.bc.o"
    )
endif ()

string(REPLACE " " ";" BMFLAGS ${CUDA_CXXFLAGS})

# Generate LLVM bitcode file with device code
# The generated file is used for link-time optimization with the JITed code
# TODO: correctly accept the includes and CXX flags
add_custom_command(OUTPUT ${CUDA_WORKDIR}/buffer-manager.bc.o
  COMMAND ${CMAKE_CXX_COMPILER}
  ${CMAKE_CURRENT_SOURCE_DIR}/lib/memory/buffer-manager.cu
  --cuda-device-only -c -emit-llvm -O3
  # TODO: correctly accept the CXX flags
  ${BMFLAGS}
  # TODO: correctly accept the includes flags
  -I${CMAKE_CURRENT_SOURCE_DIR}/include
  -isystem ${CMAKE_BINARY_DIR}/_deps/glog-download-build
  -isystem ${CMAKE_BINARY_DIR}/_deps/glog-download-src/src
  # Embed the generated file into the library
  COMMAND ld -r -b binary buffer-manager*.bc
  -o ${CUDA_WORKDIR}/buffer-manager.bc.o
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/lib/memory/buffer-manager.cu
  IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/lib/memory/buffer-manager.cu
  WORKING_DIRECTORY ${CUDA_WORKDIR}
  COMMENT "Building CUDA LLVM bitcode ${CUDA_WORKDIR}/buffer-manager.bc.o"
  )
#-----------------------------------------------------------------------------

install(DIRECTORY DESTINATION ${CMAKE_INSTALL_BINDIR}/generated_code
  #COMPONENT ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_BINARIES}
  COMPONENT ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_BINARIES}
  )

add_library(platform SHARED
  ${platform_cxx_cu}
  ${platform_cu}
  ${platform_cu_objs}
  ${platform_cxx}
  )

# If we have compiler requirements for this library, list them
# here
target_compile_features(platform PUBLIC cxx_std_20)

# Depend on a library that we defined in the top-level file
target_link_libraries(platform
  PUBLIC
  glog::glog
  NUMA::NUMA
  RDMA::RDMA
  ${CUDA_LIBS}
  cli-flags::cli-flags
  PRIVATE
  magic_enum::magic_enum
  )

if (VTUNE AND VTUNE_ENABLE)
  target_link_libraries(platform PUBLIC vtune::vtune)
endif ()

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
