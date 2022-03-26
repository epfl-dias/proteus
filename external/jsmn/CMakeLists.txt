project(jsmn VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  jsmn.c
  )

target_compile_options(${PROJECT_NAME}
  PRIVATE -Wno-declaration-after-statement
  )

include(proteus-lib)
add_library(jsmn::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
