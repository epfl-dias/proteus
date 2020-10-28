project(oltp_adaptor VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(oltp_adaptor_cxx
        aeolus-plugin.cpp
        )

foreach (target ${oltp_adaptor_cxx})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach ()

add_library(oltp_adaptor SHARED
        ${oltp_adaptor_cxx}
        )

set_target_properties(oltp_adaptor PROPERTIES
        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs")


# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(oltp_adaptor
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR})

# If we have compiler requirements for this library, list them
# here
target_compile_features(oltp_adaptor
        PUBLIC cxx_std_20)

target_link_libraries(oltp_adaptor
        PUBLIC
        oltp::engine
        codegen::olap)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS oltp_adaptor EXPORT ${PROJECT_NAME}Config
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS oltp_adaptor FILE ${PROJECT_NAME}Config.cmake)

add_library(oltp::adaptor ALIAS oltp_adaptor)

