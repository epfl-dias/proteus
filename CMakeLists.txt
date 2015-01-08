cmake_minimum_required(VERSION 2.8)
project(raw)



SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_C_COMPILER             "/usr/bin/clang")

#find_package(LLVM REQUIRED CONFIG)
LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
find_package(LLVM REQUIRED)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

find_package(Boost REQUIRED)

file(GLOB main common/*.cpp util/*.cpp values/*.cpp expressions/*.cpp plugins/*.cpp plugins/output/*.cpp operators/*.cpp)
file(GLOB jsmn jsmn/*.c)

include_directories("${PROJECT_SOURCE_DIR}")

add_library(jsmn ${jsmn})
add_library(rawlib ${main})
#target_link_libraries(raw ${llvm_libs})
add_executable(raw main.cpp)
target_link_libraries(raw rawlib)
target_link_libraries(raw jsmn)
target_link_libraries(raw -lglog)
execute_process(COMMAND llvm-config --ldflags OUTPUT_VARIABLE llvm_ldflags OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND llvm-config --libs OUTPUT_VARIABLE llvm_libs OUTPUT_STRIP_TRAILING_WHITESPACE)
target_link_libraries(raw ${llvm_ldflags} ${llvm_libs})

enable_testing()
include(ExternalProject)
ExternalProject_Add(gtest SOURCE_DIR /usr/src/gtest PREFIX /tmp/gtest INSTALL_COMMAND "")
ExternalProject_Get_Property(gtest binary_dir)
set(GTEST_LIBRARY ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_MAIN_LIBRARY ${binary_dir}/${CMAKE_FIND_LIBRARY_PREFIXES}gtest_main.a)
file(GLOB gtest_src tests/*.cpp)
include_directories("/usr/include/gtest")
add_executable(unit-tests ${gtest_src})
target_link_libraries(unit-tests rawlib)
target_link_libraries(unit-tests jsmn)
target_link_libraries(unit-tests ${GTEST_LIBRARY})
target_link_libraries(unit-tests ${GTEST_MAIN_LIBRARY})
target_link_libraries(unit-tests -lglog)
target_link_libraries(unit-tests ${llvm_ldflags} ${llvm_libs})
