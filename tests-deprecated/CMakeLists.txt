set(gtest_libs ${GTEST_MAIN} ${GTEST} ${default_libs})

foreach(target
	tests-csv
	tests-hashing
	tests-json
	tests-nest
	tests-operators
	tests-output
	tests-sailors
	tests-string
	)
	add_executable(dunit-${target} ${target}.cpp)
	target_link_libraries(dunit-${target} ${gtest_libs})
	install(TARGETS dunit-${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

# One test to rule them all
file(GLOB gtest_src *.cpp)
add_executable(dunit-tests ${gtest_src})
target_link_libraries(dunit-tests ${gtest_libs})
install(TARGETS dunit-tests
	RUNTIME DESTINATION raw
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

enable_testing()
add_test(UT dunit-tests)
