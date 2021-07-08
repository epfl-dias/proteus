project(olap_flags VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  cli-flags.cpp
  )

# Depend on a library that we defined in the top-level file
target_link_libraries(olap_flags
  PRIVATE
  proteus::olap
  PUBLIC
  cli-flags::cli-flags
  )

include(proteus-lib)
add_library(codegen::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
add_library(proteus::olap-flags ALIAS ${PROJECT_NAME})
