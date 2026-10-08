project(stack VERSION 0.1 LANGUAGES CXX)

# Pure, regular C++ files
add_library(${PROJECT_NAME} SHARED
  q01.cpp
  q04.cpp
  q06.cpp
  q11.cpp
  q12.cpp
  q14.cpp
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
