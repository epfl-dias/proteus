project(olap-perf-main VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  lib/main.cpp
  )

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  proteus::olap
  proteus::olap-flags
  )

proteus_target_link_perftest_libraries(${PROJECT_NAME} proteus::olap lib/main.cpp)

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
