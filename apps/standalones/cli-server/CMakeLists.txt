# Set the name of the app from the folder name
get_filename_component(APP_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(proteus${APP_NAME} VERSION 0.1 LANGUAGES CXX)

add_executable(${PROJECT_NAME} cli-server.cpp)

target_link_libraries(${PROJECT_NAME}
	codegen::olap_flags
	proteus::olap
	proteus::command_provider
)

include(proteus-app)
