project(ch100w VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  q01.sql.json.cpp
  q04.sql.json.cpp
  q06.sql.json.cpp
  q12.sql.json.cpp
  q19.sql.json.cpp
  )

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  proteus::olap
  oltp::adaptor
  )

include(proteus-lib)
add_library(queries::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
