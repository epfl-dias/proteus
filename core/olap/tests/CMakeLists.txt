add_definitions(-DTEST_INPUTS="inputs")
add_definitions(-DTEST_OUTPUTS="outputs")

add_subdirectory(olap-test-environment)
add_subdirectory(olap-test-main)
add_subdirectory(olap-test-utils)

add_subdirectory(queue-test)
add_subdirectory(plugin-tests)
add_subdirectory(operator-tests)
