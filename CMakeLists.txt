cmake_minimum_required(VERSION 2.8)
project(raw)

SET (CMAKE_CXX_COMPILER "/usr/bin/clang++")
SET (CMAKE_C_COMPILER "/usr/bin/clang")

# slightly complicated way to link against LLVM, the Ubuntu cmake module file
# is wrong and we have to hack something instead.
find_program(LLVM_CONFIG "llvm-config")
if(${LLVM_CONFIG} STREQUAL "LLVM_CONFIG-NOTFOUND")
    message(FATAL_ERROR "LLVM not found")
endif()
execute_process(COMMAND llvm-config --cppflags OUTPUT_VARIABLE LLVM_CPPFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --includedir OUTPUT_VARIABLE LLVM_INCLUDE_DIRS OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --ldflags OUTPUT_VARIABLE llvm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --libs OUTPUT_VARIABLE llvm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)

find_package(Boost REQUIRED)

# manually build jsmn
file(GLOB jsmn jsmn/*.c)
add_library(jsmn ${jsmn})

include_directories("${PROJECT_SOURCE_DIR}")


# code generation library
file(GLOB codegen common/*.cpp util/*.cpp values/*.cpp expressions/*.cpp plugins/*.cpp plugins/output/*.cpp operators/*.cpp)
add_library(codegen ${codegen})

# not too sure if this is not a test file in fact
add_executable(rawmain main.cpp)
target_link_libraries(rawmain codegen)
target_link_libraries(rawmain jsmn)
target_link_libraries(rawmain -lglog)
target_link_libraries(rawmain ${llvm_ldflags} ${llvm_libs})

# build googletest lib using the source code provided by the ubuntu package
include(ExternalProject)
ExternalProject_Add(gtest SOURCE_DIR /usr/src/gtest PREFIX /tmp/gtest INSTALL_COMMAND "")
ExternalProject_Get_Property(gtest binary_dir)
set(GTEST_LIBRARY ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_MAIN_LIBRARY ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.a)

file(GLOB gtest_src tests/*.cpp)
add_executable(unit-tests ${gtest_src})
target_link_libraries(unit-tests codegen)
target_link_libraries(unit-tests jsmn)
target_link_libraries(unit-tests ${GTEST_LIBRARY})
target_link_libraries(unit-tests ${GTEST_MAIN_LIBRARY})
target_link_libraries(unit-tests -lglog)
target_link_libraries(unit-tests ${llvm_ldflags} ${llvm_libs})

# data files
file(COPY inputs DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

set_target_properties(codegen rawmain unit-tests PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
