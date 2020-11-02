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

# Additional (to the proteus-lib default includes) include directories
target_include_directories(${PROJECT_NAME}
    SYSTEM PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        )

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        proteus::olap
    PRIVATE
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        proteus::command_provider
        proteus::protos)

include(proteus-lib)
add_library(proteus::runtime ALIAS ${PROJECT_NAME})
