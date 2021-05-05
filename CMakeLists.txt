cmake_minimum_required(VERSION 3.17.3)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/modules")
include(proteus-prolog)

#-----------------------------------------------------------------------
# Which binaries to build:
#-----------------------------------------------------------------------

# Tests
# not too sure if we should build "main", it looks like it has the same
# test cases as the gtests
set(UNIT_TESTS TRUE)
set(DUNIT_TESTS FALSE)

# Benchmarks
set(TPCH FALSE)
set(REALWORLD_EXPERIMENTS FALSE)

#-----------------------------------------------------------------------
# Setup environment for the project
#-----------------------------------------------------------------------
# Flag to control whether we should build the dependencies
option(STANDALONE "If ON, installs required subprojects. Otherwise the parent project should provide them" ON)

# Whether to enable or not VTune support, if available
set(VTUNE_ENABLE TRUE)

# FIXME: Next command is used until we move the binaries to the bin folder
set(CMAKE_INSTALL_BINDIR pelago)

# Generates only a few package, one per group, named:
set(PROTEUS_CPACK_COMP_SUFFIX_DEV "dev")
set(PROTEUS_CPACK_COMP_SUFFIX_BINARIES "bin")
set(PROTEUS_CPACK_COMP_SUFFIX_TESTS "tests")
set(PROTEUS_CPACK_COMP_SUFFIX_BENCHMARKS "bench")
set(PROTEUS_CPACK_COMP_SUFFIX_EXPERIMENTS "bench")

# Main packages shared between multiple components.
set(PROTEUS_CPACK_COMP_DEV "proteus-${PROTEUS_CPACK_COMP_SUFFIX_DEV}")
set(PROTEUS_CPACK_COMP_BINARIES "proteus-${PROTEUS_CPACK_COMP_SUFFIX_BINARIES}")
set(PROTEUS_CPACK_COMP_TESTS "proteus-${PROTEUS_CPACK_COMP_SUFFIX_TESTS}")
set(PROTEUS_CPACK_COMP_BENCHMARKS "proteus-${PROTEUS_CPACK_COMP_SUFFIX_BENCHMARKS}")
set(PROTEUS_CPACK_COMP_EXPERIMENTS "proteus-${PROTEUS_CPACK_COMP_SUFFIX_EXPERIMENTS}")

#-----------------------------------------------------------------------
if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif (POLICY CMP0077)

#-----------------------------------------------------------------------
# Now that the compilers are set, define project name and version
project(proteus VERSION 2.0 LANGUAGES C CXX)

# This has to be done after enabling languages
include(GNUInstallDirs)
include(GetPrerequisites)

#-----------------------------------------------------------------------
# Setup link flags
#-----------------------------------------------------------------------
# Manage correctly Library path
# Per https://gitlab.kitware.com/cmake/community/-/wikis/doc/cmake/RPATH-handling#always-full-rpath

# use, i.e. don't skip the full RPATH for the build tree
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
if ("${isSystemDir}" STREQUAL "-1")
  # LSC: Make path relative to executable, so it works from build
  #      or install directories, without having to set
  #      LD_LIBRARY_PATH.
  set(CMAKE_INSTALL_RPATH "\$ORIGIN/../lib")
endif ("${isSystemDir}" STREQUAL "-1")

#-----------------------------------------------------------------------
# Setup Compilation flags
#-----------------------------------------------------------------------
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

set(CMAKE_CXX_STANDARD 17)
# Otherwise the -std=gnu++XX is used instead of the -std=c++XX, as one
# would expect from the property above
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CXX_STANDARD_REQUIRED TRUE)

#-----------------------------------------------------------------------
# Sanitize to comply with CMP0004
# For CMAKE variables, we can use string(STRIP "" var) for this

# Place each function and data in its own section
string(STRIP "${COMPILER_FLAGS} -ffunction-sections -fdata-sections" COMPILER_FLAGS)

# By default, tune for the local machine architecture
string(STRIP "${COMPILER_FLAGS} -march=native -mtune=native" COMPILER_FLAGS)

# Hide inline methods by default
string(STRIP "${COMPILER_FLAGS} -fvisibility-inlines-hidden" COMPILER_FLAGS)

# Currently our code fails to link for -O0, -O1
string(STRIP "${COMPILER_FLAGS} -O3" COMPILER_FLAGS)

#-----------------------------------------------------------------------
# Add the compiler flags
string(STRIP "${CMAKE_CXX_FLAGS} ${COMPILER_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} ${COMPILER_FLAGS}" CMAKE_C_FLAGS)

# Add utilities to enable the warning flags
include(warning-flags)

#-----------------------------------------------------------------------
# Various dependencies
include(llvm-virtual)
include(cuda)
include(rapidjson)
include(gflags)
include(glog)
include(gtest)
include(grpc)
include(cuckoo)
include(vtune)

#-----------------------------------------------------------------------
# Build Documentation
#-----------------------------------------------------------------------
find_package(Doxygen)
if (DOXYGEN_FOUND)
  doxygen_add_docs(
    doxygen
    ${CMAKE_CURRENT_SOURCE_DIR}/apps
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    COMMENT "Generate doxygen pages"
  )
endif ()

install(DIRECTORY "frontends/R/" DESTINATION ${CMAKE_INSTALL_PREFIX}/pelago/frontends/R/)	# Creates <..>/pelago/frontends/R

#-----------------------------------------------------------------------
# Build our libraries
#-----------------------------------------------------------------------

# build libjsmn
add_subdirectory(external/jsmn)

# code generation library
add_subdirectory(core/platform)
add_subdirectory(core/planner)
add_subdirectory(core/olap)
add_subdirectory(core/oltp)
add_subdirectory(core/htap)
add_subdirectory(runtime)

#-----------------------------------------------------------------------
# Build our executables
#-----------------------------------------------------------------------
add_subdirectory(apps)
add_subdirectory(tests)

#-----------------------------------------------------------------------
# Package definitions
#-----------------------------------------------------------------------
include(InstallRequiredSystemLibraries)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${${PROJECT_NAME}_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${${PROJECT_NAME}_VERSION_MINOR}")

# Build separate package by default
set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
#set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
set(CPACK_DEB_COMPONENT_INSTALL ON)

set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS ON)

set(CPACK_ARCHIVE_THREADS 0) # Use all cpu threads for compression, when supported

include(CPack)

cpack_add_component(${PROTEUS_CPACK_COMP_BINARIES}
  DISPLAY_NAME Binaries
  DESCRIPTION "Main daemon and tools"
  REQUIRED
  BOLD_TITLE
  )

cpack_add_component(${PROTEUS_CPACK_COMP_DEV}
  DISPLAY_NAME Development
  DESCRIPTION "Header files, library archives"
  DISABLED
  BOLD_TITLE
  )

cpack_add_component(${PROTEUS_CPACK_COMP_TESTS}
  DISPLAY_NAME Tests
  DESCRIPTION "Tests"
  DISABLED
  BOLD_TITLE
  )

cpack_add_component(${PROTEUS_CPACK_COMP_BENCHMARKS}
  DISPLAY_NAME Benchmarks
  DESCRIPTION "Benchmarks"
  DISABLED
  BOLD_TITLE
  )

#cpack_add_component(${PROTEUS_CPACK_COMP_EXPERIMENTS}
#	DISPLAY_NAME Experiments
#	DESCRIPTION  "Experiments"
#	DISABLED
#	BOLD_TITLE
#)
