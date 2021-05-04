project(cuckoo VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features (cuckoo INTERFACE cxx_std_17)

target_include_directories(${PROJECT_NAME} SYSTEM
  INTERFACE
  $<BUILD_INTERFACE:${libcuckoo_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

set(_proteus_install_target ${PROJECT_NAME})

include(_proteus-install-lib)

add_library(cuckoo::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
