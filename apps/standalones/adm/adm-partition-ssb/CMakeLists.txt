find_package(nvcomp)
if (nvcomp_FOUND)
  message(STATUS "nvcomp found, configuring ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Set the name of the app from the folder name
get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Include all source files, should be changed to a manual list to be safer.
file(GLOB APP_SRCS CONFIGURE_DEPENDS main.cpp)

project(proteus${APP_NAME} VERSION 0.1 LANGUAGES CXX)
add_executable(${PROJECT_NAME} ${APP_SRCS}
  rfor-compression.cpp
  rfor-compression.hpp
  flbp-compression.cpp
  flbp-compression.hpp
  ${CMAKE_SOURCE_DIR}/apps/standalones/adm/adm-encode-bench/kernel-fused/fastlanes/src/pack.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE
  ${CMAKE_SOURCE_DIR}/apps/standalones/adm/adm-encode-bench/kernel-fused/fastlanes/src/include)

# Depending on the olap libraries
target_link_libraries(${PROJECT_NAME}
  jemalloc
  nvcomp::nvcomp
  codegen::olap_flags
  proteus::olap
  proteus::storage
  lz4_static
)

include(proteus-experiment)
