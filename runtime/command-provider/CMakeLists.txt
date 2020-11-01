project(command_provider VERSION 0.1 LANGUAGES CXX)

add_library(command_provider SHARED
    lib/command-provider.cpp
    lib/local-command-provider.cpp
        )


#set_target_properties(olap PROPERTIES
#        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs ")

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(command_provider
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

# If we have compiler requirements for this library, list them
# here
#target_compile_features(olap_flags
#        PUBLIC cxx_std_17)

# Depend on a library that we defined in the top-level file
target_link_libraries(command_provider
    PRIVATE
        proteus::olap
        codegen::olap_flags
    PUBLIC
        cli-flags::cli-flags)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS command_provider EXPORT ${PROJECT_NAME}Targets
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
export(TARGETS command_provider FILE ${PROJECT_NAME}Config.cmake)

add_library(proteus::command_provider ALIAS command_provider)
#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?
