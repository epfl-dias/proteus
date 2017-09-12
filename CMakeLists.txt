#-----------------------------------------------------------------------
# Setup environment for the project
#-----------------------------------------------------------------------
cmake_minimum_required(VERSION 2.8)
project(raw)

# Make sure the libraries are found...
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# Build with the install RPATH already
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Set the installation path of the libraries
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

#-----------------------------------------------------------------------
# various dependencies (e.g., glog) may not be available through repos,
# esp. on the servers
find_library(GLOG NAMES libglog glog)
if(${GLOG} STREQUAL "GLOG-NOTFOUND")
    message(FATAL_ERROR "Glog not found!")
endif()

find_library(GTEST NAMES libgtest gtest)
if(${GTEST} STREQUAL "GTEST-NOTFOUND")
    message(FATAL_ERROR "GoogleTest not found!")
endif()

# list(APPEND CMAKE_CXX_SOURCE_FILE_EXTENSIONS cu)


# look for Boost related compilation flags
find_package(Boost REQUIRED)

#-----------------------------------------------------------------------
# look for LLVM related compilation flags slightly complicated way, the
# Ubuntu cmake module file is wrong and we have to hack something instead.
find_program(LLVM_CONFIG "llvm-config")
if(${LLVM_CONFIG} STREQUAL "LLVM_CONFIG-NOTFOUND")
    message(FATAL_ERROR "LLVM not found")
endif()

find_program(CLANG_CXX_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
find_program(CLANG_C_COMPILER "clang")
set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}")

# set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD 11)

# LLVM APIs we use
#set(LLVM_COMPONENTS engine executionengine analysis lto mc mcjit nativecodegen orcjit passes)
execute_process(COMMAND ${CMAKE_INSTALL_PREFIX}/bin/llvm-config --cxxflags OUTPUT_VARIABLE LLVM_CXXFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${CMAKE_INSTALL_PREFIX}/bin/llvm-config --cflags OUTPUT_VARIABLE LLVM_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${CMAKE_INSTALL_PREFIX}/bin/llvm-config --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${CMAKE_INSTALL_PREFIX}/bin/llvm-config --ldflags OUTPUT_VARIABLE LLVM_LDFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${CMAKE_INSTALL_PREFIX}/bin/llvm-config --libs ${LLVM_COMPONENTS} OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${CMAKE_INSTALL_PREFIX}/bin/llvm-config --system-libs ${LLVM_COMPONENTS} OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)

#-----------------------------------------------------------------------
include_directories(
	"${PROJECT_SOURCE_DIR}"
	"${PROJECT_SOURCE_DIR}/codegen"
	"${PROJECT_SOURCE_DIR}/multigpu/src"
	"${LLVM_INCLUDE_DIRS}"
	"${CMAKE_INSTALL_PREFIX}/include"
	)
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

# Define path to installed resources
add_definitions(-DTEST_INPUTS="${CMAKE_INSTALL_PREFIX}/raw/inputs")
add_definitions(-DTEST_OUTPUTS="${CMAKE_INSTALL_PREFIX}/raw/outputs")

#set(GCC_COMPILE_FLAGS "-march=core-avx2 -msse4 -O0 -ftree-vectorize -march=native -mtune=native -g -ggdb -fno-omit-frame-pointer")
set(WARNING_FLAGS "-Wno-unknown-warning-option -Wno-unused-variable -Wno-reorder -Wno-ignored-qualifiers -Wno-deprecated")
set(LLVM_CXXFLAGS "-x cuda -D__STRICT_ANSI__ -stdlib=libstdc++ ${LLVM_CXXFLAGS} ${WARNING_FLAGS} ${GCC_COMPILE_FLAGS} -I/usr/local/cuda/include -I/usr/local/cuda/nvvm/include")
set(LLVM_CFLAGS "${LLVM_CFLAGS} ${WARNING_FLAGS} ${GCC_COMPILE_FLAGS}")

set(CUDA_CXXFLAGS -Xcompiler -fPIC ${CUDA_CXXFLAGS} ${GCC_COMPILE_FLAGS} --expt-relaxed-constexpr --std=c++11 -arch=sm_61 -I/opt/intel/vtune_amplifier_xe_2017/include -I/usr/local/cuda/include -I/usr/local/cuda/nvvm/include)
set(CUDA_CFLAGS "${CUDA_CFLAGS} ${GCC_COMPILE_FLAGS}")


set(default_libs codegen multigpu -lnuma -lnvToolsExt -lglog ${LLVM_LIBS} ${CUDA_LIBRARIES} -lstdc++ ${LLVM_SYSTEM_LIBS} ${LLVM_LDFLAGS} -L/usr/local/cuda/lib64 -L/usr/local/cuda/nvvm/lib64 -L/usr/local/cuda/nvml/lib64 -lcuda -lcudart -lnvidia-ml -lcudadevrt -ldl -lrt -lnvvm -pthread -I/usr/local/cuda/nvvm/include -L/opt/intel/vtune_amplifier_xe_2017/lib64 -littnotify)
set(gtest_libs -lgtest libgtest_main.a ${default_libs})

#-----------------------------------------------------------------------
# Build our libraries
#-----------------------------------------------------------------------

# build libjsmn
add_subdirectory(jsmn)

# code generation library
add_subdirectory(codegen)
add_subdirectory(multigpu)

#-----------------------------------------------------------------------
# Build our executables
#-----------------------------------------------------------------------
if(TRUE)
# not too sure if we should build "main", it looks like it has the same
# test cases as the gtests
foreach(target
	main
	main-caches
	main-nest
	main-str
	main-reduce
	)
	add_executable(raw${target} ${target}.cpp)
	set_target_properties(raw${target} PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
	target_link_libraries(raw${target} ${default_libs})
	install(TARGETS raw${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

# Get rawmain to compile even if there are expected tautological comparisons
set_target_properties(rawmain PROPERTIES COMPILE_FLAGS "${LLVM_CXXFLAGS} -Wno-tautological-compare")

add_executable(plan-parsing plan-parsing.cpp)
set_target_properties(plan-parsing PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
target_link_libraries(plan-parsing ${default_libs})
endif()

if(TRUE)
add_subdirectory(benchmarks/tpch)
endif()

#real-world experiments
if(TRUE)
add_subdirectory(experiments/realworld-queries)
add_subdirectory(experiments/realworld-symantec)
add_subdirectory(experiments/realworld-vldb)

#sigmod versions
add_subdirectory(experiments/realworld)
endif()

# compile our unit-tests
add_subdirectory(tests)
add_subdirectory(tests-deprecated)

#-----------------------------------------------------------------------
# Copy data files too in the destination folder
#-----------------------------------------------------------------------
install(DIRECTORY "inputs" DESTINATION ${CMAKE_INSTALL_PREFIX}/raw)			# Creates <..>/raw/inputs
install(DIRECTORY "testResults/" DESTINATION ${CMAKE_INSTALL_PREFIX}/raw/outputs)	# Creates <..>/raw/outputs
