# Set the name of the app from the folder name
get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(proteus${APP_NAME} VERSION 0.1 LANGUAGES CXX)
add_executable(${PROJECT_NAME}
	ndp.cpp
	ndp-cli-flags.cpp
	lib/ndp-common.cpp
	lib/cluster_command_provider.cpp
	lib/example-plans/alternative-paths.cpp
	lib/example-plans/scheduling-paths.cpp
	lib/example-plans/task-description.cpp
	lib/example-plans/task-offloading.cpp
)

# Depending on the olap libraries
target_link_libraries(${PROJECT_NAME}
	codegen::olap_flags
	proteus::runtime
	proteus::plan-parser
	proteus::command_provider
)

# Add paths to the local pseudo-library
target_include_directories(${PROJECT_NAME}
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}
)

include(proteus-experiment)
