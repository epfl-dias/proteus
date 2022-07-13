get_filename_component(TS_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${TS_NAME} VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  lib/main.cpp
  )

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  proteus::storage
  )

# Add the gtest libraries, as this is a "test" library
proteus_target_link_test_libraries(${PROJECT_NAME})

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
