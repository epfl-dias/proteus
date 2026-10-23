project(proteus-generate-cb2t-dataset VERSION 0.1 LANGUAGES CXX)

add_executable(${PROJECT_NAME}
  main.cpp
  rfor-compression.cpp
  rfor-compression.hpp
  flbp-compression.cpp
  flbp-compression.hpp
  fastlanes/src/pack.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/fastlanes/include)

target_link_libraries(${PROJECT_NAME}
  jemalloc
  nvcomp::nvcomp
  codegen::olap_flags
  proteus::olap
  proteus::storage
  lz4_static)

include(proteus-experiment)
