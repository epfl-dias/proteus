project(ssb100_bloom VERSION 0.1 LANGUAGES CXX)

add_library(${PROJECT_NAME} SHARED
	q1_1.sql.json.cpp
	q1_2.sql.json.cpp
	q1_3.sql.json.cpp
	q2_1.sql.json.cpp
	q2_2.sql.json.cpp
	q2_3.sql.json.cpp
	q3_1.sql.json.cpp
	q3_2.sql.json.cpp
	q3_3.sql.json.cpp
	q3_4.sql.json.cpp
	q4_1.sql.json.cpp
	q4_2.sql.json.cpp
	q4_3.sql.json.cpp
)

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
	PUBLIC
		proteus::olap
)

include(proteus-lib)
add_library(queries::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
