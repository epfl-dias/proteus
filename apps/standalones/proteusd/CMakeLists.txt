project(proteusd VERSION 0.1 LANGUAGES CXX)

add_executable(${PROJECT_NAME} proteusd.cpp)

target_link_libraries(${PROJECT_NAME}
	codegen::olap_flags
	proteus::olap
	proteus::command_provider
)

include(proteus-app)
