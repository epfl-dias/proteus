project(cli-flags VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} INTERFACE)

target_link_libraries(${PROJECT_NAME}
  PUBLIC INTERFACE
  gflags::gflags
  )

include(proteus-headers-lib)

add_library(cli-flags::cli-flags ALIAS ${PROJECT_NAME})

# There are multiple definitions for gflags, so we need to specify which
# one we are interested in.
export(TARGETS gflags_shared APPEND FILE ${PROJECT_NAME}Config.cmake)

install(EXPORT gflags-targets DESTINATION lib/cmake/gflags)
