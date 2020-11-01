project(platform VERSION 0.1 LANGUAGES CXX)

find_package(NUMA REQUIRED)
find_package(RDMA REQUIRED)

# Pure, regular C++ files
set(platform_cxx
    common/common.cpp
    common/gpu/gpu-common.cpp
    memory/block-manager.cpp
    memory/memory-allocator.cpp
    memory/memory-manager.cpp
    network/infiniband/devices/ib.cpp
    network/infiniband/private/ib_impl.cpp
    network/infiniband/infiniband-client.cpp
    network/infiniband/infiniband-handler.cpp
    network/infiniband/infiniband-manager.cpp
    network/infiniband/infiniband-server.cpp
    storage/mmap-file.cpp
    storage/storage-manager.cpp
    topology/affinity_manager.cpp
    topology/device-manager.cpp
    topology/topology.cpp
    util/percentile.cpp
    util/profiling.cpp
    util/time-registry.cpp
    util/radix/aggregations/radix-aggr.cpp
    util/radix/joins/radix-join.cpp
    )

# C++ files requiring CUDA headers
set(platform_cxx_cu
#    operators/gpu/gpu-partitioned-hash-join-chained-helpers.cpp
    )

# Pure CUDA files
set(platform_cu
# LSC: 30.11.2018
#       Work around issue in:
#         /usr/local/cuda/targets/x86_64-linux/include/thrust/system/cuda/detail/cub/util_debug.cuh 117-136
#       Which fails with va_printf undefined.
    #util/sort/gpu/sort-util.cu

# LSC: 30.11.2018
#       For now we build two times this, as otherwise we get linking and/or
#       runtime error as LLVM cannot yet generate relocatable ptx code.
        memory/buffer-manager.cu
        util/sort/gpu/sort-util.cu
    )

foreach(target ${platform_cu})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach()

foreach(target ${platform_cxx_cu} ${platform_cu})
    set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach()

#-----------------------------------------------------------------------------
# LSC: Hopefully we will soon be able to remove this section, where we call
#      straight to NVCC
set(CUDA_WORKDIR ${CMAKE_CURRENT_BINARY_DIR}/cubin)
file(MAKE_DIRECTORY ${CUDA_WORKDIR})

if(CUDA_FOUND)
    set(platform_cu_objs
        "${CUDA_WORKDIR}/buffer-manager.bc.o"
    )
endif()

string(REPLACE " " ";" BMFLAGS ${CUDA_CXXFLAGS})

# Generate LLVM bitcode file with device code
# The generated file is used for link-time optimization with the JITed code
# TODO: correctly accept the includes and CXX flags
add_custom_command(OUTPUT ${CUDA_WORKDIR}/buffer-manager.bc.o
    COMMAND ${CMAKE_CXX_COMPILER}
        ${CMAKE_CURRENT_SOURCE_DIR}/memory/buffer-manager.cu
        --cuda-device-only -c -emit-llvm -O3
        # TODO: correctly accept the CXX flags
        ${BMFLAGS}
        # TODO: correctly accept the includes flags
        -I${CMAKE_CURRENT_SOURCE_DIR} -isystem ${CMAKE_BINARY_DIR}/_deps/glog-download-build -isystem ${CMAKE_BINARY_DIR}/_deps/glog-download-src/src
    # Embed the generated file into the library
    COMMAND ld -r -b binary buffer-manager*.bc
        -o ${CUDA_WORKDIR}/buffer-manager.bc.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/memory/buffer-manager.cu
    IMPLICIT_DEPENDS CXX ${CMAKE_CURRENT_SOURCE_DIR}/memory/buffer-manager.cu
    WORKING_DIRECTORY ${CUDA_WORKDIR}
    COMMENT "Building CUDA LLVM bitcode ${CUDA_WORKDIR}/buffer-manager.bc.o"
)
#-----------------------------------------------------------------------------


install(DIRECTORY DESTINATION pelago/generated_code)

add_library(platform SHARED
        ${platform_cxx_cu}
        ${platform_cu}
        ${platform_cu_objs}
        ${platform_cxx}
        )

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(platform
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
            $<INSTALL_INTERFACE:.>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})

# If we have compiler requirements for this library, list them
# here
target_compile_features(platform
        PUBLIC cxx_std_20)

# Depend on a library that we defined in the top-level file
target_link_libraries(platform
        PUBLIC
            glog::glog
            NUMA::NUMA
            RDMA::RDMA
            -lc++abi
	    -lrt
            ${CUDA_LIBS}
            cli-flags)

if(VTUNE AND VTUNE_ENABLE)
target_link_libraries(platform PUBLIC vtune::vtune)
endif()

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS platform EXPORT ${PROJECT_NAME}Targets
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Targets
        FILE ${PROJECT_NAME}Targets.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})

include(GNUInstallDirs)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        ${CMAKE_BINARY_DIR}/cmake/platform-config-version.cmake
        COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
        ${CMAKE_SOURCE_DIR}/cmake/platform-config.cmake.in
        ${CMAKE_BINARY_DIR}/cmake/platform-config.cmake
        INSTALL_DESTINATION share/${PROJECT_NAME}/cmake
)

install(
        FILES
        ${CMAKE_BINARY_DIR}/cmake/platform-config.cmake
        ${CMAKE_BINARY_DIR}/cmake/platform-config-version.cmake
        DESTINATION lib/cmake/${PROJECT_NAME}
)

# This makes the project importable from the build directory
export(TARGETS platform FILE ${PROJECT_NAME}Config.cmake)

add_library(proteus::platform ALIAS platform)
#enable_testing()
#add_test(UT platform-tests) # Add platform-specific tests?
