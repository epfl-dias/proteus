project(oltp_adaptor VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
	aeolus-plugin.cpp
)

# If we have compiler requirements for this library, list them
# here
target_compile_features(${PROJECT_NAME}
	PUBLIC
		cxx_std_20
)

target_link_libraries(${PROJECT_NAME}
	PUBLIC
		oltp::engine
		proteus::olap
)

include(proteus-lib)
add_library(oltp::adaptor ALIAS ${PROJECT_NAME})
