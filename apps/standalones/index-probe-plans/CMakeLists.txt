# Set the name of the app from the folder name
get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Include all source files, should be changed to a manual list to be safer.
file(GLOB APP_SRCS CONFIGURE_DEPENDS *.cpp)

project(proteus${APP_NAME} VERSION 0.1 LANGUAGES CXX)
add_executable(${PROJECT_NAME} ${APP_SRCS})

# Depending on the olap libraries
target_link_libraries(${PROJECT_NAME}
	codegen::olap_flags
	proteus::olap
	oltp::engine
	oltp::adaptor
)

include(proteus-experiment)
