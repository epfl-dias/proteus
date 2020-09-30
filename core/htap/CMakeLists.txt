project(htap VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
set(htap_cxx
	utils/utils.cpp
    )

foreach(target ${htap_cxx})
    set_source_files_properties(${target} PROPERTIES LANGUAGE CXX)
endforeach()

# foreach(target ${htap_cxx})
#     set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
# endforeach()

add_library(htap_engine SHARED
        ${htap_cxx}
        )

set_target_properties(htap_engine PROPERTIES
        LINK_FLAGS "-Wl,--no-allow-shlib-undefined -Wl,-z,defs")
# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(htap_engine
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR})

# If we have compiler requirements for this library, list them
# here
target_compile_features(htap_engine
        PUBLIC cxx_std_17)

# Depend on a library that we defined in the top-level file
target_link_libraries(htap_engine
        PUBLIC
            ${llvm_libs}
            codegen::olap
            aeolus::engine
            aeolus::plugin
            )

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS htap_engine EXPORT ${PROJECT_NAME}Config
        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
export(TARGETS htap_engine FILE ${PROJECT_NAME}Config.cmake)

add_library(htap::engine ALIAS htap_engine)
#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?

# FIXME: remove after old warnings are fixed
target_compile_options(
      htap_engine
    PRIVATE  # Not PUBLIC to at least avoid leaking the issue across libs
      "-Wno-error=suggest-override"
      "-Wno-error=suggest-destructor-override"
)
