get_filename_component(STANDALONE_DIR ${CMAKE_CURRENT_SOURCE_DIR} NAME)

project(${STANDALONE_DIR} VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
	lib/plan-parser.cpp
)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
		proteus::olap
)

include(proteus-lib)
add_library(proteus::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
