project(command_provider VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
	lib/command-provider.cpp
	lib/local-command-provider.cpp
)

target_link_libraries(${PROJECT_NAME}
	PRIVATE
		proteus::olap
		codegen::olap_flags
	PUBLIC
		cli-flags::cli-flags)

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
