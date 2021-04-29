project(jsmn VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  jsmn.c
  )

include(proteus-lib)
add_library(jsmn::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
