cmake_minimum_required(VERSION 2.8)
project(raw)

find_program(CLANG_CXX_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "${CLANG_CXX_COMPILER}")
find_program(CLANG_C_COMPILER "clang")
set(CMAKE_C_COMPILER "${CLANG_C_COMPILER}")

include_directories("${PROJECT_SOURCE_DIR}")

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

set(GCC_COMPILE_FLAGS "-march=core-avx2 -msse4 -O0 -ftree-vectorize -march=native -mtune=native -g -ggdb -fno-omit-frame-pointer")
set( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COMPILE_FLAGS}" )

set(default_libs codegen jsmn -lglog ${llvm_ldflags} -ltinfo ${llvm_libs})
set(gtest_libs -lgtest libgtest_main.a ${default_libs})

# build libjsmn
file(GLOB jsmn jsmn/*.c)
add_library(jsmn ${jsmn})

# code generation library
file(GLOB codegen common/*.cpp util/*.cpp util/radix/*.cpp util/radix/joins/*.cpp util/radix/aggregations/*.cpp values/*.cpp expressions/*.cpp plugins/*.cpp plugins/output/*.cpp operators/*.cpp memory/*.cpp plan/*.cpp postgres/*.cpp)
add_library(codegen ${codegen})
set_target_properties(codegen PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})

#if(FALSE)
# not too sure if we should build "main", it looks like it has the same test cases as the gtests
add_executable(rawmain main.cpp)
set_target_properties(rawmain PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(rawmain ${default_libs})

add_executable(rawmain-caches main-caches.cpp)
set_target_properties(rawmain-caches PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(rawmain-caches ${default_libs})

add_executable(rawmain-nest main-nest.cpp)
set_target_properties(rawmain-nest PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(rawmain-nest ${default_libs})

add_executable(rawmain-str main-str.cpp)
set_target_properties(rawmain-str PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(rawmain-str ${default_libs})

add_executable(rawmain-reduce main-reduce.cpp)
set_target_properties(rawmain-reduce PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(rawmain-reduce ${default_libs})
#endif()

add_subdirectory(benchmarks/tpch)

#real-world experiments
add_subdirectory(experiments/realworld-vldb)
add_subdirectory(experiments/realworld-symantec)

#sigmod versions
add_executable(spam-bin experiments/realworld/spam-bin.cpp)
set_target_properties(spam-bin PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-bin ${default_libs})

add_executable(spam-csv experiments/realworld/spam-csv.cpp)
set_target_properties(spam-csv PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-csv ${default_libs})

add_executable(spam-json experiments/realworld/spam-json.cpp)
set_target_properties(spam-json PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-json ${default_libs})

add_executable(spam-bin-csv experiments/realworld/spam-bin-csv.cpp)
set_target_properties(spam-bin-csv PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-bin-csv ${default_libs})

add_executable(spam-bin-json experiments/realworld/spam-bin-json.cpp)
set_target_properties(spam-bin-json PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-bin-json ${default_libs})

add_executable(spam-csv-json experiments/realworld/spam-csv-json.cpp)
set_target_properties(spam-csv-json PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-csv-json ${default_libs})

add_executable(spam-bin-csv-json experiments/realworld/spam-bin-csv-json.cpp)
set_target_properties(spam-bin-csv-json PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-bin-csv-json ${default_libs})

add_executable(spam-bin-csv-json-v2 experiments/realworld/spam-bin-csv-json-v2.cpp)
set_target_properties(spam-bin-csv-json-v2 PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-bin-csv-json-v2 ${default_libs})

add_executable(spam-csv-cached experiments/realworld/spam-csv-cached.cpp)
set_target_properties(spam-csv-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-csv-cached ${default_libs})

add_executable(spam-csv-cached-columnar experiments/realworld/spam-csv-cached-columnar.cpp)
set_target_properties(spam-csv-cached-columnar PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-csv-cached-columnar ${default_libs})

add_executable(spam-json-cached experiments/realworld/spam-json-cached.cpp)
set_target_properties(spam-json-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-json-cached ${default_libs})


file(GLOB spam_all experiments/realworld-queries/*.cpp)
#add_executable(spam-workload experiments/spam-workload.cpp)
add_executable(spam-workload ${spam_all})
set_target_properties(spam-workload PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-workload ${default_libs})

add_executable(plan-parsing plan-parsing.cpp)
set_target_properties(plan-parsing PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(plan-parsing ${default_libs})

# compile our unit-tests
add_subdirectory(tests)

# copy data files too
file(COPY "inputs" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "testResults" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
