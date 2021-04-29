# Include all source files, should be changed to a manual list to be safer.
file(GLOB APP_SRCS CONFIGURE_DEPENDS *.cpp)

project(spam_workload VERSION 0.1 LANGUAGES CXX)

add_executable(${PROJECT_NAME} ${APP_SRCS})

target_link_libraries(${PROJECT_NAME}
  proteus::olap
  )

include(proteus-experiment)
