cmake_minimum_required(VERSION 3.5.1)

project(protos VERSION 0.1 LANGUAGES CXX)

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
get_filename_component(hw_proto "clustercontrolplane.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)


# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/proto/clustercontrolplane.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/proto/clustercontrolplane.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/proto/clustercontrolplane.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/proto/clustercontrolplane.grpc.pb.h")
add_custom_command(
        OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/proto/"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}/proto/"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}/proto/"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
        DEPENDS "${hw_proto}")

# If we have compiler requirements for this library, list them
# here
add_library(${PROJECT_NAME} SHARED
        ${hw_proto_srcs}
        ${hw_grpc_srcs})
target_include_directories(${PROJECT_NAME}
    SYSTEM PUBLIC INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/proto>
        $<INSTALL_INTERFACE:proto>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

target_compile_features(${PROJECT_NAME}
    PRIVATE cxx_std_20)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${_REFLECTION}
        ${_GRPC_GRPCPP}
        ${_PROTOBUF_LIBPROTOBUF})

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

add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
#enable_testing()
#add_test(UT proteus-runtime-tests) # Add proteus-runtime-specific tests?
