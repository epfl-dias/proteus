cmake_minimum_required(VERSION 3.5.1)

project(distributed-runtime VERSION 0.1 LANGUAGES CXX)

find_package(Threads REQUIRED)

message(STATUS "Using gRPC via add_subdirectory (FetchContent).")
include(FetchContent)
FetchContent_Declare(
        grpc
        GIT_REPOSITORY https://github.com/grpc/grpc.git
        GIT_TAG       v1.33.1)
FetchContent_MakeAvailable(grpc)

# Since FetchContent uses add_subdirectory under the hood, we can use
# the grpc targets directly from this build.
set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
if(CMAKE_CROSSCOMPILING)
    find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin)
else()
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
endif()

# Proto file
get_filename_component(hw_proto "protos/clustercontrolplane.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)


# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/clustercontrolplane.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/clustercontrolplane.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/clustercontrolplane.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/clustercontrolplane.grpc.pb.h")
add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}")

#
#
#
#set(PROTO_F
#        clustercontrolplane
#        )
#
#foreach(PROTO_FILE ${PROTO_F})
#    # Proto file
#    get_filename_component(${PROTO_F}_name "protos/${PROTO_FILE}.proto" ABSOLUTE)
#    get_filename_component(${PROTO_F}_path "${PROTO_F}_name" PATH)
#
#    # ${CMAKE_CURRENT_BINARY_DIR}
#
#    # Generated sources
#    set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.pb.cc")
#    set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.pb.h")
#    set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.grpc.pb.cc")
#    set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.grpc.pb.h")
#
#    set(PROTO_SRC_FILES "${PROTO_SRC_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.pb.cc")
#    set(PROTO_HDR_FILES "${PROTO_HDR_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.pb.h")
#    set(GRPC_SRC_FILES "${GRPC_SRC_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.grpc.pb.cc")
#    set(GRPC_HDR_FILES "${GRPC_HDR_FILES} ${CMAKE_CURRENT_BINARY_DIR}/${PROTO_FILE}.grpc.pb.h")
#
#    add_custom_command(
#            OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
#            COMMAND ${_PROTOBUF_PROTOC}
#            ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#            --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#            -I "${PROTO_F}_path"
#            --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#            "${PROTO_F}_name"
#            DEPENDS "${PROTO_F}_name")
#
#endforeach(PROTO_FILE)
#-----------------------------------------------------------------------
# Proteus: Distributed Runtime
#-----------------------------------------------------------------------

# Pure, regular C++ files
set(distributed_runtime_cxx
    ${hw_proto_srcs}
    ${hw_grpc_srcs}
        lib/cluster-manager.cpp
        lib/cluster-control.cpp
        )

add_library(distributed-runtime SHARED ${distributed_runtime_cxx})


# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(distributed-runtime
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
target_compile_features(distributed-runtime
        PUBLIC cxx_std_20)


# Depend on a library that we defined in the top-level file
target_link_libraries(distributed-runtime
    PUBLIC
        codegen::olap
    PRIVATE
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS distributed-runtime EXPORT ${PROJECT_NAME}Targets
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
export(TARGETS distributed-runtime FILE ${PROJECT_NAME}Config.cmake)

add_library(proteus::runtime ALIAS distributed-runtime)
#enable_testing()
#add_test(UT proteus-runtime-tests) # Add proteus-runtime-specific tests?
