cmake_minimum_required(VERSION 2.8)
project(raw)

find_program(CLANG_CXX_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
find_program(CLANG_C_COMPILER "clang")
set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/codegen")

include_directories("${CMAKE_INSTALL_PREFIX}/include")
link_directories("${CMAKE_INSTALL_PREFIX}/lib")

# look for LLVM related compilation flags slightly complicated way, the Ubuntu
# cmake module file is wrong and we have to hack something instead.
find_program(LLVM_CONFIG "llvm-config")
if(${LLVM_CONFIG} STREQUAL "LLVM_CONFIG-NOTFOUND")
    message(FATAL_ERROR "LLVM not found")
endif()
execute_process(COMMAND llvm-config --cppflags OUTPUT_VARIABLE LLVM_CPPFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --ldflags OUTPUT_VARIABLE llvm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --libs OUTPUT_VARIABLE llvm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)

# various dependencies (e.g., glog) may not be available through repos - esp. on the servers
find_library(GLOG NAMES libglog glog)
if(${GLOG} STREQUAL "GLOG-NOTFOUND")
    message(FATAL_ERROR "Glog not found!")
endif()

find_library(GTEST NAMES libgtest gtest)
if(${GTEST} STREQUAL "GTEST-NOTFOUND")
    message(FATAL_ERROR "GoogleTest not found!")
endif()

# look for Boost related compilation flags
find_package(Boost REQUIRED)

# Define path to installed resources
add_definitions(-DTEST_INPUTS="${CMAKE_INSTALL_PREFIX}/raw/inputs")
add_definitions(-DTEST_OUTPUTS="${CMAKE_INSTALL_PREFIX}/raw/outputs")

set(GCC_COMPILE_FLAGS "-march=core-avx2 -msse4 -O0 -ftree-vectorize -march=native -mtune=native -g -ggdb -fno-omit-frame-pointer")
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}" )

set(default_libs codegen jsmn -lglog ${llvm_ldflags} -ltinfo ${llvm_libs})
set(gtest_libs -lgtest libgtest_main.a ${default_libs})

# build libjsmn
file(GLOB jsmn jsmn/*.c)
add_library(jsmn ${jsmn})

# code generation library
add_subdirectory(codegen)

#if(FALSE)
# not too sure if we should build "main", it looks like it has the same test cases as the gtests
foreach(target
	main
	main-caches
	main-nest
	main-str
	main-reduce
	)
	add_executable(raw${target} ${target}.cpp)
	set_target_properties(raw${target} PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
	target_link_libraries(raw${target} ${default_libs})
endforeach(target)
#endif()

add_executable(plan-parsing plan-parsing.cpp)
set_target_properties(plan-parsing PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(plan-parsing ${default_libs})

add_subdirectory(benchmarks/tpch)

#real-world experiments
add_subdirectory(experiments/realworld-queries)
add_subdirectory(experiments/realworld-symantec)
add_subdirectory(experiments/realworld-vldb)

#sigmod versions
add_subdirectory(experiments/realworld)

# compile our unit-tests
add_subdirectory(tests)

# copy data files too
install(DIRECTORY "inputs" DESTINATION ${CMAKE_INSTALL_PREFIX}/raw)			# Creates <..>/raw/inputs
install(DIRECTORY "testResults/" DESTINATION ${CMAKE_INSTALL_PREFIX}/raw/outputs)	# Creates <..>/raw/outputs
