project(olap-test-utils VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
  lib/test-utils.cpp
  )

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  proteus::olap
  )

# Add the gtest libraries, as this is a "test" library
proteus_target_link_test_libraries(${PROJECT_NAME})

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
