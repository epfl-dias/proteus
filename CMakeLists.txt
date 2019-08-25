cmake_minimum_required(VERSION 3.10.2)

# Set LLVM suffix for systems with multiple LLVM installed to request
# a specific version using -DLLVM_VERSION_SUFFIX=7

# Flag to control whether we should build the dependencies
option(STANDALONE "If ON, installs required subprojects. Otherwise the parent project should provide them" ON)

#-----------------------------------------------------------------------
# Start by looking for the compilers.
# This should be done before any project(.) or enable_language(.) calls

# Look for LLVM related compilation flags in a slightly complicated way,
# the Ubuntu cmake module file is incorrect for our use case.

if (DEFINED LLVM_VERSION_SUFFIX)
	set(LLVM_VERSION_SUFFIX "-${LLVM_VERSION_SUFFIX}")
endif()

# NOTE: If we find llvm-config, we assume clang and clang++ are visible
#       in the PATH
find_program(LLVM_CONFIG "llvm-config${LLVM_VERSION_SUFFIX}")
if(${LLVM_CONFIG} STREQUAL "LLVM_CONFIG-NOTFOUND")
	message(FATAL_ERROR "LLVM not found")
endif()

find_program(CLANG_CXX_COMPILER "clang++${LLVM_VERSION_SUFFIX}")
set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
find_program(CLANG_C_COMPILER "clang${LLVM_VERSION_SUFFIX}")
set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}")

if (POLICY CMP0048)
	cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

# Now that the compilers are set, define project name and version
project(aeolus VERSION 0.1)

#-----------------------------------------------------------------------
# Which binaries to build:
#-----------------------------------------------------------------------
# Simply comment the lines of this section to skip some of the binaries
# to build.
set(MAINS server)

# Tests
# not too sure if we should build "main", it looks like it has the same
# test cases as the gtests
set(UNIT_TESTS TRUE)

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

set(CMAKE_CXX_STANDARD 17)
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

# Turn on maximum code compliance and all the warnings
string(STRIP "${WARNING_FLAGS} -pedantic -Weverything" WARNING_FLAGS)

# Turn warnings into errors
string(STRIP "${WARNING_FLAGS} -Werror" WARNING_FLAGS)
# Disable warning-as-error for deprecated calls
# string(STRIP "${WARNING_FLAGS} -Wno-error=deprecated" WARNING_FLAGS)

# Our Code generates the following warnings:
string(STRIP "${WARNING_FLAGS} -Wno-unused-macros -Wno-assign-enum" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-c++98-compat" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-c++98-compat-pedantic" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-cast-align" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-cast-qual" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-conditional-uninitialized" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-conversion" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-covered-switch-default" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-deprecated" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-double-promotion" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-exit-time-destructors" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-global-constructors" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-ignored-qualifiers" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-implicit-fallthrough" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-missing-prototypes" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-missing-variable-declarations" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-old-style-cast" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-padded" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-reorder"  WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-reserved-id-macro" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shadow" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shadow-field" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shadow-field-in-constructor" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-shorten-64-to-32" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-sign-compare" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-sign-conversion" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-switch-enum" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-tautological-compare" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-undefined-func-template" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unreachable-code" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unreachable-code-break" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unreachable-code-return" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unused-command-line-argument" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unused-parameter" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-unused-variable" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-vla" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-vla-extension" WARNING_FLAGS)
string(STRIP "${WARNING_FLAGS} -Wno-weak-vtables" WARNING_FLAGS)

# Unit-tests:
string(STRIP "${WARNING_FLAGS} -Wno-shift-sign-overflow" WARNING_FLAGS)



## OLD ONES
string(STRIP "${WARNING_FLAGS} -Wno-unused-function -Wno-c++2a-compat -Wno-atomic-implicit-seq-cst -Wno-extra-semi-stmt -Wno-unused-template" WARNING_FLAGS)

#-----------------------------------------------------------------------
# Add the compiler flags
string(STRIP "${CMAKE_CXX_FLAGS} ${COMPILER_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} ${COMPILER_FLAGS}" CMAKE_C_FLAGS)

# LLVM APIs we use
set(LLVM_COMPONENTS "aggressiveinstcombine all all-targets analysis asmparser asmprinter binaryformat bitreader bitwriter codegen core coroutines coverage debuginfocodeview debuginfodwarf debuginfomsf debuginfopdb demangle dlltooldriver engine executionengine fuzzmutate globalisel instcombine instrumentation inteljitevents interpreter ipo irreader libdriver lineeditor linker lto mc mcdisassembler mcjit mcparser mirparser native nativecodegen nvptx nvptxasmprinter nvptxcodegen nvptxdesc nvptxinfo objcarcopts object objectyaml option orcjit passes profiledata runtimedyld scalaropts selectiondag support symbolize tablegen target transformutils vectorize windowsmanifest x86 x86asmparser x86asmprinter x86codegen x86desc x86disassembler x86info x86utils")
set(LLVM_COMPONENTS "")
execute_process(COMMAND ${LLVM_CONFIG} --libs ${LLVM_COMPONENTS} OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND ${LLVM_CONFIG} --includedir ${LLVM_COMPONENTS} OUTPUT_VARIABLE LLVM_INCLUDEDIR OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${LLVM_CONFIG} --libdir ${LLVM_COMPONENTS} OUTPUT_VARIABLE LLVM_LINKDIR OUTPUT_STRIP_TRAILING_WHITESPACE)

include_directories(
	SYSTEM ${LLVM_INCLUDEDIR}
)

link_directories(
	${LLVM_LINKDIR}
)

#-----------------------------------------------------------------------
# Various dependencies
#-----------------------------------------------------------------------
# CUDA
find_package(CUDA)
if(CUDA_FOUND)
	# TODO: the CUDA package has been deprecated and replaced by native support for the language.
	# 		We should consider replacing it with the correct usage of enable_language(CUDA),
	# 		CMAKE_CUDA_COMPILER, CMAKE_CUDA_FLAGS etc
	if(DEFINED _NVCC_ARCHS)
		list(REMOVE_DUPLICATES _NVCC_ARCHS)

		CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_CU_ARCHS ${_NVCC_ARCHS})
	else(DEFINED _NVCC_ARCHS)
		CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_CU_ARCHS Auto)
	endif(DEFINED _NVCC_ARCHS)

	# Create arguments for nvcc and clang
	string(REGEX REPLACE " " [[;]] CUDA_CU_ARCHS_readable "${CUDA_CU_ARCHS_readable}")
	foreach(line ${CUDA_CU_ARCHS_readable})
		# Clang always includes PTX in the generated library for forward compatibility.
		# On the other hand, in nvcc it's optional.
		# So, if CUDA_CU_ARCHS contains a directive to include the PTX we should ignore it,
		# otherwise, clang complains.
		# DO NOT remove them from nvcc's flags!

		# Architecture names that start with "compute_" are nvcc directives to generate PTX
		if (NOT ${line} MATCHES "compute_*")
			list(APPEND CUDA_CXX_ARCHS "--cuda-gpu-arch=${line}")
		endif()
	endforeach(line)

	# C++ & CUDA hybrid files
	string(STRIP "${CUDA_CXXFLAGS} -Wno-format-pedantic" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} -x cuda" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} --cuda-path=${CUDA_TOOLKIT_ROOT_DIR}" CUDA_CXXFLAGS)
	# Enable llvm::*->dump()
	string(STRIP "${CUDA_CXXFLAGS} -DLLVM_ENABLE_DUMP" CUDA_CXXFLAGS)

	# Convert the list to a string
	string(REGEX REPLACE [[;]] " " CUDA_CXX_ARCHS "${CUDA_CXX_ARCHS}")
	string(STRIP "${CUDA_CXXFLAGS} ${CUDA_CXX_ARCHS}" CUDA_CXXFLAGS)
	string(STRIP "${CUDA_CXXFLAGS} -DMAXRREGCOUNT=32" CUDA_CXXFLAGS)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -DMAXRREGCOUNT=32 --maxrregcount=32)

	# Because of how the "command" function works, we have to use a
	# CMAKE list, and not a string for CUDA_CUFLAGS

	# CUDA-only flags, used with nvcc
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -isystem ${CMAKE_INSTALL_PREFIX}/include)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -isystem ${LLVM_INCLUDEDIR})
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -I ${PROJECT_SOURCE_DIR})
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -I ${PROJECT_SOURCE_DIR}/codegen)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} ${CUDA_CU_ARCHS})
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} --std=c++11)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -Xcompiler -fPIC)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} --expt-relaxed-constexpr)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} -O3)
	set(CUDA_CUFLAGS ${CUDA_CUFLAGS} --generate-line-info -line-info)

	set(CUDA_LIBS -lnvToolsExt -lcuda -lcudart -lnvidia-ml -lcudadevrt -lnvvm)

	include_directories(
		SYSTEM ${CUDA_INCLUDE_DIRS}
		)

	link_directories(
		"${CUDA_TOOLKIT_ROOT_DIR}/lib64"
		"${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs"
		"${CUDA_TOOLKIT_ROOT_DIR}/nvvm/lib64"
		)

	set(CUDA "CUDA-FOUND")
else()
	message("Warning: Building without support for GPUs (nvcc not detected)")

	set(CUDA_CXXFLAGS -x c++ -DNCUDA)
	set(CUDA_CUFLAGS "")
	set(CUDA_LIBS "")

	set(CUDA "CUDA-NOTFOUND")
endif()

# GFlags
if (TRUE)
	# Add google test here, so that we do not propagate to it the WARNING_FLAGS
	configure_file(CMakeLists.txt.gflags.in gflags-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
		RESULT_VARIABLE result
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gflags-download )
	if(result)
		message(FATAL_ERROR "CMake step for gflags failed: ${result}")
	endif()

	execute_process(COMMAND ${CMAKE_COMMAND} --build .
		RESULT_VARIABLE result
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/gflags-download )
	if(result)
		message(FATAL_ERROR "Build step for gflags failed: ${result}")
	endif()

	# Add gflags directly to our build. This defines
	# the gtest and gtest_main targets.
	add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/gflags-src
		${CMAKE_CURRENT_BINARY_DIR}/gflags-build
		EXCLUDE_FROM_ALL)
endif()

#-----------------------------------------------------------------------
# GoogleLog
find_library(GLOG NAMES glog)
if(${GLOG} STREQUAL "GLOG-NOTFOUND")
	message(FATAL_ERROR "Glog not found!")
endif()

#-----------------------------------------------------------------------
# GoogleTest
if (STANDALONE)
	# Add google test here, so that we do not propagate to it the WARNING_FLAGS
	configure_file(CMakeLists.txt.gtest.in googletest-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
		RESULT_VARIABLE result
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
		message(FATAL_ERROR "CMake step for googletest failed: ${result}")
	endif()

	execute_process(COMMAND ${CMAKE_COMMAND} --build .
		RESULT_VARIABLE result
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
		message(FATAL_ERROR "Build step for googletest failed: ${result}")
	endif()

	# Add googletest directly to our build. This defines
	# the gtest and gtest_main targets.
	add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
		${CMAKE_CURRENT_BINARY_DIR}/googletest-build
		EXCLUDE_FROM_ALL)

	set(GTEST gtest)
	set(GTEST_MAIN gtest_main)
else()
	find_library(GTEST NAMES gtest)
	if(${GTEST} STREQUAL "GTEST-NOTFOUND")
		message(FATAL_ERROR "GoogleTest not found!")
	endif()

	find_library(GTEST_MAIN NAMES gtest_main)
	if(${GTEST_MAIN} STREQUAL "GTEST_MAIN-NOTFOUND")
		message(FATAL_ERROR "GoogleTest main not found!")
	endif()
endif()

#-----------------------------------------------------------------------
# Setup the library list, as the order is important do it last



#-----------------------------------------------------------------------
# Setup the library list, as the order is important do it last

# `codegen` needs to be unquoted to be replaced, as CMake controls the build of
# that library
set(default_libs ${default_libs} codegen )
set(default_libs ${default_libs} aeolus_engine )
set(default_libs ${default_libs} aeolus_bench )
set(default_libs ${default_libs} -lnuma )
set(default_libs ${default_libs} ${LLVM_LIBS} -lLLVMSupport)
set(default_libs ${default_libs} ${CUDA_LIBS})
set(default_libs ${default_libs} gflags::gflags ${GLOG} gflags::gflags)
set(default_libs ${default_libs} -lc++abi -lpthread -lrt)


#-----------------------------------------------------------------------
# Intel VTune profile support
find_library(VTUNE ittnotify HINTS "/opt/intel/vtune_amplifier/lib64")
if(VTUNE AND VTUNE_ENABLE)
	include_directories(
		SYSTEM "/opt/intel/vtune_amplifier/include"
		)
	set(default_libs ${default_libs} ${VTUNE} -ldl)
endif()

#-----------------------------------------------------------------------
# Define path to installed resources
add_definitions(-DTEST_INPUTS="inputs")
add_definitions(-DTEST_OUTPUTS="outputs")

#-----------------------------------------------------------------------
# Define path to installed resources
#add_definitions(-DTEST_INPUTS="${CMAKE_INSTALL_PREFIX}/pelago/inputs")

#-----------------------------------------------------------------------
# Copy data files too in the destination folder
#-----------------------------------------------------------------------
#install(DIRECTORY "inputs" DESTINATION ${CMAKE_INSTALL_PREFIX}/pelago)			# Creates <..>/pelago/inputs

#-----------------------------------------------------------------------
# Build our libraries
#-----------------------------------------------------------------------

# Hash library
# https://github.com/preshing/junction.git
# https://github.com/preshing/turf.git
#add_subdirectory(lib/libcuckoo/libcuckoo)

include_directories(
	"${PROJECT_SOURCE_DIR}/engine"
	"${PROJECT_SOURCE_DIR}/benchmarks"
	SYSTEM "${PROJECT_SOURCE_DIR}/lib"
	SYSTEM "${PROTEUS_SOURCE_DIR}"
	SYSTEM "${PROTEUS_SOURCE_DIR}/codegen"

	# Let's see if we pull in headers which requires this, if not let's drop the whole block
	# FIXME: system libraries should not be included with quotes...
	# Add local (for use with ""-style includes) path to the sysroot
	"${CMAKE_INSTALL_PREFIX}/include"
	)


# Add the warning flags
string(STRIP "${CMAKE_CXX_FLAGS} ${WARNING_FLAGS}" CMAKE_CXX_FLAGS)
string(STRIP "${CMAKE_C_FLAGS} ${WARNING_FLAGS}" CMAKE_C_FLAGS)

#-----------------------------------------------------------------------
# Build our executables
#-----------------------------------------------------------------------
add_subdirectory(engine)
add_subdirectory(benchmarks)


foreach(target ${MAINS})
	add_executable(aeolus-${target} ${target}.cpp)
	target_link_libraries(aeolus-${target} ${default_libs})
	install(TARGETS aeolus-${target}
		RUNTIME DESTINATION aeolus
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

if (NOT STANDALONE)
add_subdirectory(adaptors)
endif(NOT STANDALONE)

#-----------------------------------------------------------------------
# Define path to installed resources
add_definitions(-DTEST_INPUTS="inputs")
add_definitions(-DTEST_OUTPUTS="outputs")

# Unit-tests
if(UNIT_TESTS)
	add_subdirectory(tests)
endif()
