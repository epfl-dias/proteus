project(aeolus_plugin VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(aeolus_plugin_cxx
    aeolus-plugin.cpp
    )

#-----------------------------------------------------------------------------

add_library(aeolus_plugin SHARED
        ${aeolus_plugin_cxx}
    )
target_link_libraries(aeolus_plugin aeolus_engine codegen::olap)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS aeolus_plugin EXPORT ${PROJECT_NAME}Config
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS aeolus_plugin FILE ${PROJECT_NAME}Config.cmake)

#enable_testing()
#add_test(UT aeolus_plugin-tests)
