project(aeolus_bench VERSION 0.1 LANGUAGES CXX)
# Pure, regular C++ files
set(aeolus_bench_cxx
        tpcc/tpcc_64.cpp
		tpcc/txn.cpp
		tpcc/utils.cpp
		tpcc/consistency_verifiers.cpp)

add_library(aeolus_bench SHARED
        ${aeolus_bench_cxx}
    )

# Define headers for this library. PUBLIC headers are used for
# compiling the library, and will be added to consumers' build
# paths.
target_include_directories(aeolus_bench
		PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
			$<INSTALL_INTERFACE:.>
		PRIVATE
			src)

# If we have compiler requirements for this library, list them
# here
target_compile_features(aeolus_bench
		PUBLIC cxx_std_17)

# Depend on a library that we defined in the top-level file
target_link_libraries(aeolus_bench
        aeolus_engine
		aeolus_plugin)

# 'make install' to the correct locations (provided by GNUInstallDirs).
install(TARGETS aeolus_bench EXPORT ${PROJECT_NAME}Config
		ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
		RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})  # This is for Windows

# This makes the project importable from the install directory
# Put config file in per-project dir (name MUST match), can also
# just go into 'cmake'.
#install(EXPORT ${PROJECT_NAME}Config DESTINATION share/${PROJECT_NAME}/cmake)

# This makes the project importable from the build directory
#export(TARGETS codegen FILE ${PROJECT_NAME}Config.cmake)

add_library(aeolus::bench ALIAS aeolus_bench)
#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?

