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

add_executable(tpch-csv-projections tpch-csv-projections.cpp)
set_target_properties(tpch-csv-projections PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-csv-projections ${default_libs})

add_executable(tpch-csv-projections-cached tpch-csv-projections-cached.cpp)
set_target_properties(tpch-csv-projections-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-csv-projections-cached ${default_libs})

add_executable(tpch-csv-selections tpch-csv-selections.cpp)
set_target_properties(tpch-csv-selections PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-csv-selections ${default_libs})

add_executable(tpch-csv-selections-cached tpch-csv-selections-cached.cpp)
set_target_properties(tpch-csv-selections-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-csv-selections-cached ${default_libs})

add_executable(tpch-csv-joins tpch-csv-joins.cpp)
set_target_properties(tpch-csv-joins PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-csv-joins ${default_libs})

add_executable(tpch-csv-groups tpch-csv-groups.cpp)
set_target_properties(tpch-csv-groups PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-csv-groups ${default_libs})

add_executable(tpch-json-projections tpch-json-projections.cpp)
set_target_properties(tpch-json-projections PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-projections ${default_libs})

add_executable(tpch-json-projections-cached tpch-json-projections-cached.cpp)
set_target_properties(tpch-json-projections-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-projections-cached ${default_libs})

add_executable(tpch-json-selections tpch-json-selections.cpp)
set_target_properties(tpch-json-selections PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-selections ${default_libs})

add_executable(tpch-json-selections-cached tpch-json-selections-cached.cpp)
set_target_properties(tpch-json-selections-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-selections-cached ${default_libs})

add_executable(tpch-json-groups tpch-json-groups.cpp)
set_target_properties(tpch-json-groups PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-groups ${default_libs})

add_executable(tpch-json-groups-cached tpch-json-groups-cached.cpp)
set_target_properties(tpch-json-groups-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-groups-cached ${default_libs})

add_executable(tpch-bin-projections tpch-bin-projections.cpp)
set_target_properties(tpch-bin-projections PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-projections ${default_libs})

add_executable(tpch-bin-selections tpch-bin-selections.cpp)
set_target_properties(tpch-bin-selections PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-selections ${default_libs})

add_executable(tpch-bin-joins tpch-bin-joins.cpp)
set_target_properties(tpch-bin-joins PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-joins ${default_libs})

add_executable(tpch-bin-joins-profiling tpch-bin-joins-profiling.cpp)
set_target_properties(tpch-bin-joins-profiling PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-joins-profiling ${default_libs})

add_executable(tpch-bin-joins-cached tpch-bin-joins-cached.cpp)
set_target_properties(tpch-bin-joins-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-joins-cached ${default_libs})

add_executable(tpch-bin-groups tpch-bin-groups.cpp)
set_target_properties(tpch-bin-groups PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-groups ${default_libs})

add_executable(tpch-bin-groups-profiling tpch-bin-groups-profiling.cpp)
set_target_properties(tpch-bin-groups-profiling PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-groups-profiling ${default_libs})

add_executable(tpch-bin-groups-cached tpch-bin-groups-cached.cpp)
set_target_properties(tpch-bin-groups-cached PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-bin-groups-cached ${default_libs})

add_executable(tpch-materializers tpch-materializers.cpp)
set_target_properties(tpch-materializers PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-materializers ${default_libs})
#endif()

add_executable(tpch-json-joins tpch-json-joins.cpp)
set_target_properties(tpch-json-joins PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-joins ${default_libs})

add_executable(tpch-json-unnests tpch-json-unnests.cpp)
set_target_properties(tpch-json-unnests PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(tpch-json-unnests ${default_libs})

#real-world experiments
add_executable(symantec-json symantec-json.cpp)
set_target_properties(symantec-json PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(symantec-json ${default_libs})

###################
###VLDB VERSIONS###
###################
file(GLOB spam_all_vldb experiments/realworld-vldb/*.cpp)
add_executable(spam-workload-vldb ${spam_all_vldb})
set_target_properties(spam-workload-vldb PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-workload-vldb ${default_libs})

#add_executable(spam-bin-vldb experiments/realworld-vldb/spam-bin.cpp)
#set_target_properties(spam-bin-vldb PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
#target_link_libraries(spam-bin-vldb ${default_libs})
#
#add_executable(spam-csv-cached-columnar-vldb experiments/realworld-vldb/spam-csv-cached-columnar.cpp)
#set_target_properties(spam-csv-cached-columnar-vldb PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
#target_link_libraries(spam-csv-cached-columnar-vldb ${default_libs})


#add_executable(spam-bin-csv-vldb experiments/realworld-vldb/spam-bin-csv.cpp)
#set_target_properties(spam-bin-csv-vldb PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
#target_link_libraries(spam-bin-csv-vldb ${default_libs})

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
file(GLOB gtest_src tests/*.cpp)
add_executable(unit-tests ${gtest_src})
set_target_properties(unit-tests PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(unit-tests ${gtest_libs})

file(GLOB gtest_src_joins tests/tests-sailors.cpp)
add_executable(unit-tests-joins ${gtest_src_joins})
set_target_properties(unit-tests-joins PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(unit-tests-joins ${gtest_libs})

file(GLOB gtest_src_json tests/tests-json.cpp)
add_executable(unit-tests-json ${gtest_src_json})
set_target_properties(unit-tests-json PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(unit-tests-json ${gtest_libs})

file(GLOB gtest_src_output tests/tests-output.cpp)
add_executable(unit-tests-output ${gtest_src_output})
set_target_properties(unit-tests-output PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(unit-tests-output ${gtest_libs})

file(GLOB gtest_src_plans tests/tests-plan-parsing.cpp)
add_executable(unit-tests-plan-parsing ${gtest_src_plans})
set_target_properties(unit-tests-plan-parsing PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(unit-tests-plan-parsing ${gtest_libs})

# copy data files too
file(COPY "inputs" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY "testResults" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

enable_testing()
add_test(UT unit-tests)
