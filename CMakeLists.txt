cmake_minimum_required(VERSION 2.8)
project(raw)

#-----------------------------------------------------------------------
# Which binaries to build:
#-----------------------------------------------------------------------
# Simply comment the lines of this section to skip some of the binaries
# to build.
set(MAINS main-server) # Interface to the executor

# Tests
# not too sure if we should build "main", it looks like it has the same
# test cases as the gtests
set(MAINS ${MAINS} main main-caches main-nest main-str main-reduce)
set(PLAN_PARSING TRUE)
set(UNIT_TESTS TRUE)

# Benchmarks
set(TPCH TRUE)
set(REALWORLD_EXPERIMENTS TRUE)

#-----------------------------------------------------------------------
# Setup environment for the project
#-----------------------------------------------------------------------
# Make sure the libraries are found...
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# Build with the install RPATH already
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Set the installation path of the libraries
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_CXX_STANDARD 14)
# Otherwise the -std=gnu++XX is used instead of the -std=c++XX, as one
# would expect from the property above
set(CMAKE_CXX_EXTENSIONS FALSE)
set(CXX_STANDARD_REQUIRED TRUE)

# Whether to enable or not VTune support, if available
set(VTUNE_ENABLE TRUE)

#-----------------------------------------------------------------------
# Sanitize to comply with CMP0004
# For CMAKE variables, we can use string(STRIP "" var) for this

# Setup Compilation and link flags
#-----------------------------------------------------------------------
# Turn on maximum code compliance and all the warnings
string(STRIP "${COMPILER_FLAGS} -pedantic -Weverything" COMPILER_FLAGS)

# Place each function and data in its own section
string(STRIP "${COMPILER_FLAGS} -ffunction-sections -fdata-sections" COMPILER_FLAGS)

# By default, tune for the local machine architecture
string(STRIP "${COMPILER_FLAGS} -march=native -mtune=native" COMPILER_FLAGS)

# Hide inline methods by default
string(STRIP "${COMPILER_FLAGS} -fvisibility-inlines-hidden" COMPILER_FLAGS)

# By default add debug symbols, we are not yet ready for production releases
string(STRIP "${COMPILER_FLAGS} -g" COMPILER_FLAGS)

# Currently our code fails to link for -O0, -O1
string(STRIP "${COMPILER_FLAGS} -O2" COMPILER_FLAGS)

# Turn warnings into errors
string(STRIP "${WARNING_FLAGS} -Werror" WARNING_FLAGS)

# LLVM Code generates the following warnings
string(STRIP "${WARNING_FLAGS} -Wno-unused-parameter" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-padded" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-missing-noreturn" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shadow-field-in-constructor" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shadow" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-conversion" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-redundant-parens" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-old-style-cast" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-undef" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-documentation-unknown-command" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-comma" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-switch-enum" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-signed-enum-bitfield" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-missing-variable-declarations" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-extra-semi" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-undefined-func-template" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-weak-vtables" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-attributes" WARNING_FLAGS)

# CUDA Code generates the following warnings
string(STRIP "${WARNING_FLAGS} -Wno-documentation-deprecated-sync" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-documentation" WARNING_FLAGS)

# Our Code generates the following warnings:
string(STRIP "${WARNING_FLAGS} -Wno-unused-variable" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unused-private-field" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-reorder"  WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-ignored-qualifiers" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-deprecated" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-cast-qual" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-sign-compare" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-sign-conversion" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-vla-extension" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unused-command-line-argument" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-c++98-compat-pedantic" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-c++98-compat" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-assign-enum" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-reserved-id-macro" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shorten-64-to-32" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-zero-as-null-pointer-constant" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-header-hygiene" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-exit-time-destructors" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-global-constructors" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-missing-prototypes" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-newline-eof" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unreachable-code" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unreachable-code-return" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-conditional-uninitialized" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-covered-switch-default" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-double-promotion" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-cast-align" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shadow-field" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-used-but-marked-unused" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unreachable-code-break" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-implicit-fallthrough" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unused-macros" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-vla" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-tautological-compare" WARNING_FLAGS)

# Unit-tests:
string(STRIP "${WARNING_FLAGS} -Wno-shift-sign-overflow" WARNING_FLAGS)

#-----------------------------------------------------------------------
# Add the compiler flags
string(STRIP "${CMAKE_CXX_FLAGS} ${COMPILER_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} ${COMPILER_FLAGS}" CMAKE_C_FLAGS)

# Add the warning flags
string(STRIP "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} ${WARNING_FLAGS}" CMAKE_C_FLAGS)

include_directories(
	"${PROJECT_SOURCE_DIR}"
	"${PROJECT_SOURCE_DIR}/codegen"
	# FIXME: system libraries should not be included with quotes...
	# Add local (for use with ""-style includes) path to the sysroot
	"${CMAKE_INSTALL_PREFIX}/include"
	)

#-----------------------------------------------------------------------
# Look for LLVM related compilation flags in a slightly complicated way,
# the Ubuntu cmake module file is incorrect for our use case.
# NOTE: If we find llvm-config, we assume clang and clang++ are visible
#       in the PATH
find_program(LLVM_CONFIG "llvm-config")
if(${LLVM_CONFIG} STREQUAL "LLVM_CONFIG-NOTFOUND")
    message(FATAL_ERROR "LLVM not found")
endif()

find_program(CLANG_CXX_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
find_program(CLANG_C_COMPILER "clang")
set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}")

# LLVM APIs we use
set(LLVM_COMPONENTS "aggressiveinstcombine all all-targets analysis asmparser asmprinter binaryformat bitreader bitwriter codegen core coroutines coverage debuginfocodeview debuginfodwarf debuginfomsf debuginfopdb demangle dlltooldriver engine executionengine fuzzmutate globalisel instcombine instrumentation inteljitevents interpreter ipo irreader libdriver lineeditor linker lto mc mcdisassembler mcjit mcparser mirparser native nativecodegen nvptx nvptxasmprinter nvptxcodegen nvptxdesc nvptxinfo objcarcopts object objectyaml option orcjit passes profiledata runtimedyld scalaropts selectiondag support symbolize tablegen target transformutils vectorize windowsmanifest x86 x86asmparser x86asmprinter x86codegen x86desc x86disassembler x86info x86utils")
set(LLVM_COMPONENTS "")
execute_process(COMMAND ${CMAKE_INSTALL_PREFIX}/bin/llvm-config --libs ${LLVM_COMPONENTS} OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

#-----------------------------------------------------------------------
# Various dependencies
#-----------------------------------------------------------------------
# CUDA
find_program(HAS_NVCC nvcc)
if(HAS_NVCC)
	# C++ & CUDA hybrid files
	string(STRIP "${CUDA_CXXFLAGS} -Wno-format-pedantic" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} -x cuda" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} --cuda-gpu-arch=sm_61" CUDA_CXXFLAGS)

	# Because of how the "command" function works, we have to use a
	# CMAKE list, and not a string for CUDA_CUFLAGS

	# CUDA-only flags, used with nvcc
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -isystem ${CMAKE_INSTALL_PREFIX}/include)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -I ${PROJECT_SOURCE_DIR})
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -I ${PROJECT_SOURCE_DIR}/codegen)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -arch=sm_61 --std=c++11)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -Xcompiler -fPIC)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} --expt-relaxed-constexpr)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -O3)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} --generate-line-info -line-info)

	set(CUDA_LIBS -lnvToolsExt -lcuda -lcudart -lnvidia-ml -lcudadevrt -lnvvm)

	include_directories(
		"/usr/local/cuda/include"
		)

	link_directories(
		"/usr/local/cuda/lib64"
		"/usr/local/cuda/lib64/stubs"
		"/usr/local/cuda/nvvm/lib64"
		)

	set(CUDA "CUDA-FOUND")
else()
	message("Warning: Building without support for GPUs (nvcc not detected)")

	set(CUDA_CXXFLAGS -x c++ -DNCUDA)
	set(CUDA_CUFLAGS "")
	set(CUDA_LIBS "")

	set(CUDA "CUDA-NOTFOUND")
endif()

#-----------------------------------------------------------------------
# GoogleLog
find_library(GLOG NAMES glog)
if(${GLOG} STREQUAL "GLOG-NOTFOUND")
	message(FATAL_ERROR "Glog not found!")
endif()

#-----------------------------------------------------------------------
# GoogleTest
find_library(GTEST NAMES gtest)
if(${GTEST} STREQUAL "GTEST-NOTFOUND")
	message(FATAL_ERROR "GoogleTest not found!")
endif()

find_library(GTEST_MAIN NAMES gtest_main)
if(${GTEST_MAIN} STREQUAL "GTEST_MAIN-NOTFOUND")
	message(FATAL_ERROR "GoogleTest main not found!")
endif()

#-----------------------------------------------------------------------
# Intel VTune profile support
find_library(VTUNE ittnotify HINTS "/opt/intel/vtune_amplifier/lib64")
if(VTUNE AND VTUNE_ENABLE)
	include_directories(
		"/opt/intel/vtune_amplifier/include"
		)
	set(default_libs ${VTUNE} -ldl ${default_libs})
endif()

#-----------------------------------------------------------------------
# Setup the library list, as the order is important do it last

# `codegen` needs to be unquoted to be replaced, as CMake controls the build of
# that library
set(default_libs ${default_libs} codegen )
set(default_libs ${default_libs} ${LLVM_LIBS} -lLLVMSupport)
set(default_libs ${default_libs} ${CUDA_LIBS})
set(default_libs ${default_libs} ${GLOG})
set(default_libs ${default_libs} -lc++abi -lpthread -lrt)

#-----------------------------------------------------------------------
# Define path to installed resources
add_definitions(-DTEST_INPUTS="${CMAKE_INSTALL_PREFIX}/raw/inputs")
add_definitions(-DTEST_OUTPUTS="${CMAKE_INSTALL_PREFIX}/raw/outputs")

#-----------------------------------------------------------------------
# Copy data files too in the destination folder
#-----------------------------------------------------------------------
install(DIRECTORY "inputs" DESTINATION ${CMAKE_INSTALL_PREFIX}/raw)			# Creates <..>/raw/inputs
install(DIRECTORY "testResults/" DESTINATION ${CMAKE_INSTALL_PREFIX}/raw/outputs)	# Creates <..>/raw/outputs

#-----------------------------------------------------------------------
# Build our libraries
#-----------------------------------------------------------------------

# build libjsmn
add_subdirectory(jsmn)

# code generation library
add_subdirectory(codegen)

#-----------------------------------------------------------------------
# Build our executables
#-----------------------------------------------------------------------
foreach(target ${MAINS})
	add_executable(raw${target} ${target}.cpp)
	target_link_libraries(raw${target} ${default_libs})
	install(TARGETS raw${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

if(PLAN_PARSING)
	add_executable(plan-parsing plan-parsing.cpp)
	target_link_libraries(plan-parsing ${default_libs})
endif()

# Unit-tests
if(UNIT_TESTS)
	add_subdirectory(tests)
	add_subdirectory(tests-deprecated)
endif()

# Benchmarks
if(TPCH)
	add_subdirectory(benchmarks/tpch)
endif()

# real-world experiments
if(REALWORLD_EXPERIMENTS)
	add_subdirectory(experiments/realworld-queries)
	add_subdirectory(experiments/realworld-symantec)
	add_subdirectory(experiments/realworld-vldb)

	# sigmod versions
	add_subdirectory(experiments/realworld)
endif()
