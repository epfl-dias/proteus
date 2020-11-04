project(olap_prepared_queries VERSION 0.1 LANGUAGES CXX)

add_subdirectory(queries/ch100w)
add_subdirectory(queries/ssb100)
add_subdirectory(queries/ssb100_bloom)
add_subdirectory(queries/ssb1000)

add_executable(${PROJECT_NAME}
	run-prepared-query.cpp # Interface to the executor
)

# Depending on the olap libraries
target_link_libraries(${PROJECT_NAME}
	proteus::olap
	codegen::olap_flags
	proteus::platform
	queries::ch100w
	queries::ssb100
	queries::ssb100_bloom
	queries::ssb1000
)

include(proteus-experiment)
