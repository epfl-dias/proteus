project(cuckoo VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
add_library(cuckoo INTERFACE)

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(cuckoo
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
            $<INSTALL_INTERFACE:.>
        )

# If we have compiler requirements for this library, list them
# here
target_compile_features(cuckoo
        INTERFACE cxx_std_17)

# Depend on a library that we defined in the top-level file
target_link_libraries(cuckoo)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS cuckoo EXPORT ${PROJECT_NAME}Config
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS cuckoo FILE ${PROJECT_NAME}Config.cmake)

#enable_testing()
#add_test(UT cuckoo-tests) # Add codegen-specific tests?
