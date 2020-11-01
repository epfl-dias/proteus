project(olap_flags VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(olap_flags_cxx
        cli-flags.cpp
        )

add_library(olap_flags SHARED
        ${olap_flags_cxx}
        )


#set_target_properties(olap PROPERTIES
#        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs ")

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(olap_flags
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
target_link_libraries(olap_flags
    PRIVATE
        proteus::olap
    PUBLIC
        cli-flags::cli-flags)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS olap_flags EXPORT ${PROJECT_NAME}Targets
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
export(TARGETS olap_flags FILE ${PROJECT_NAME}Config.cmake)

add_library(codegen::olap_flags ALIAS olap_flags)
#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?
