cmake_minimum_required(VERSION 3.5.1)

add_subdirectory(protos)

project(distributed-runtime VERSION 0.1 LANGUAGES CXX)

#-----------------------------------------------------------------------
# Proteus: Distributed Runtime
#-----------------------------------------------------------------------

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)

# Pure, regular C++ files
set(distributed_runtime_cxx
        lib/cluster-manager.cpp
        lib/cluster-control.cpp
        )

add_library(${PROJECT_NAME} SHARED ${distributed_runtime_cxx})


# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(${PROJECT_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    SYSTEM PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        )

# If we have compiler requirements for this library, list them
# here

target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_20)

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
    PUBLIC
        proteus::olap
    PRIVATE
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        proteus::command_provider
        proteus::protos)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS ${PROJECT_NAME} EXPORT ${PROJECT_NAME}Targets
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

# This makes the project importable from the build directory
export(TARGETS ${PROJECT_NAME} FILE ${PROJECT_NAME}Config.cmake)

add_library(proteus::runtime ALIAS ${PROJECT_NAME})
#enable_testing()
#add_test(UT proteus-runtime-tests) # Add proteus-runtime-specific tests?
