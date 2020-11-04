project(htap_engine VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
	query-interface.cpp
)

# If we have compiler requirements for this library, list them
# here
target_compile_features(${PROJECT_NAME}
	PUBLIC
		cxx_std_20
)

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
	PUBLIC
		proteus::olap
		oltp::engine
		oltp::adaptor
)

include(proteus-lib)
add_library(htap::engine ALIAS ${PROJECT_NAME})
