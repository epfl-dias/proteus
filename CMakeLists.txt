cmake_minimum_required(VERSION 2.8)
project(raw)

find_program(CLANG_CXX_COMPILER "clang++")
SET (CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
find_program(CLANG_C_COMPILER "clang")
SET (CMAKE_C_COMPILER "${CLANG_C_COMPILER}")

# look for Boost related compilation flags
find_package(Boost REQUIRED)

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

# build libjsmn
file(GLOB jsmn jsmn/*.c)
add_library(jsmn ${jsmn})

include_directories("${PROJECT_SOURCE_DIR}")

# various dependencies (e.g., glog) may not be available through repos - esp. on the servers
include_directories("$ENV{HOME}/include")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} )
find_library(GLOG NAMES libglog glog HINTS "$ENV{HOME}/lib")
if(${GLOG} STREQUAL "GLOG-NOTFOUND")
    message(FATAL_ERROR "Glog not found")
endif()
set(glog_ldflags "-L$ENV{HOME}/lib")


# code generation library
file(GLOB codegen common/*.cpp util/*.cpp values/*.cpp expressions/*.cpp plugins/*.cpp plugins/output/*.cpp operators/*.cpp memory/*.cpp)
add_library(codegen ${codegen})
set_target_properties(codegen PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})

# not too sure if we should build "main", it looks like it has the same test cases as the gtests
add_executable(rawmain main.cpp)
set_target_properties(rawmain PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(rawmain codegen)
target_link_libraries(rawmain jsmn)
target_link_libraries(rawmain -lglog ${glog_ldflags})
target_link_libraries(rawmain ${llvm_ldflags} ${llvm_libs})

# build googletest lib using the source code provided by the ubuntu package.
include(ExternalProject)
if(GTEST)
 if(NOT EXISTS ${GTEST})
  message(FATAL_ERROR "location " ${GTEST} " doesn't exist")
 endif()
 include_directories(${GTEST}/include)
else()
 set(GTEST /usr/src/gtest)
endif()

find_path(GTEST_SRC NAMES src/gtest_main.cc PATHS ${GTEST})

if(${GTEST_SRC} STREQUAL "GTEST_SRC-NOTFOUND")
 message(FATAL_ERROR "src/gtest_main.cc not found in ${GTEST}")
endif()

ExternalProject_Add(gtest SOURCE_DIR ${GTEST_SRC} PREFIX ${CMAKE_CURRENT_BINARY_DIR} INSTALL_COMMAND "")
ExternalProject_Get_Property(gtest binary_dir)
set(GTEST_LIBRARY ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_MAIN_LIBRARY ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.a)

# compile our unit-tests
file(GLOB gtest_src tests/*.cpp)
add_executable(unit-tests ${gtest_src})
set_target_properties(unit-tests PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(unit-tests codegen)
target_link_libraries(unit-tests jsmn)
target_link_libraries(unit-tests ${GTEST_LIBRARY})
target_link_libraries(unit-tests ${GTEST_MAIN_LIBRARY})
target_link_libraries(unit-tests -lglog ${glog_ldflags})
target_link_libraries(unit-tests ${llvm_ldflags} ${llvm_libs})
# copy data files too
file(COPY "inputs" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()
add_test(UT unit-tests)
