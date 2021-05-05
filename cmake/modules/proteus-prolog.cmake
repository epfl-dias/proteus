#-----------------------------------------------------------------------
# Start by looking for the compilers.
# This should be done before any project(.) or enable_language(.) calls

# Look for LLVM related compilation flags in a slightly complicated way,
# the Ubuntu cmake module file is incorrect for our use case.

# Set LLVM suffix for systems with multiple LLVM installed to request
# a specific version using -DLLVM_VERSION_SUFFIX=7
if (DEFINED LLVM_VERSION_SUFFIX)
  set(LLVM_REQUIRED_VERSION "${LLVM_VERSION_SUFFIX}")
  set(LLVM_VERSION_SUFFIX "-${LLVM_VERSION_SUFFIX}")
else ()
  set(LLVM_REQUIRED_VERSION 11)
endif ()

# FIXME: Should we use find_package and ${LLVM_TOOLS_BINARY_DIR} to find clang?
find_program(CLANG_CXX_COMPILER "clang++${LLVM_VERSION_SUFFIX}" REQUIRED)
set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
find_program(CLANG_C_COMPILER "clang${LLVM_VERSION_SUFFIX}" REQUIRED)
set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}")

function(FetchContent name)
  # Check if population has already been performed
  FetchContent_GetProperties(${name})
  string(TOLOWER "${name}" lcName)
  if (NOT ${lcName}_POPULATED)
    # Fetch the content using previously declared details
    FetchContent_Populate(${name})

    # Bring the populated content into the build
    add_subdirectory(${${lcName}_SOURCE_DIR} ${${lcName}_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif ()
endfunction()
