project(ssb VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
add_library(${PROJECT_NAME} SHARED
  q1_1.cpp
  q1_2.cpp
  q1_3.cpp
  q2_1.cpp
  q2_2.cpp
  q2_3.cpp
  q3_1.cpp
  q3_2.cpp
  q3_3.cpp
  q3_4.cpp
  q4_1.cpp
  q4_2.cpp
  q4_3.cpp
  stats.cpp
  )

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  proteus::olap
  proteus::query-shaping
  )

include(proteus-lib)
add_library(queries::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
