project(aeolus_plugin VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(aeolus_plugin_cxx
        aeolus-plugin.cpp
        )

foreach (target ${aeolus_plugin_cxx})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach ()

add_library(aeolus_plugin SHARED
        ${aeolus_plugin_cxx}
        )

set_target_properties(aeolus_plugin PROPERTIES
        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs")
# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(aeolus_plugin
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

# If we have compiler requirements for this library, list them
# here
target_compile_features(aeolus_plugin
        PUBLIC cxx_std_17)

target_link_libraries(aeolus_plugin
        PUBLIC
        aeolus::engine
        codegen::olap)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS aeolus_plugin EXPORT ${PROJECT_NAME}Config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS aeolus_plugin FILE ${PROJECT_NAME}Config.cmake)

add_library(aeolus::plugin ALIAS aeolus_plugin)
#enable_testing()
#add_test(UT aeolus_plugin-tests)
