foreach(target
	tests-joins
	tests-json
	tests-cpu-ssb
	tests-gpu-ssb
	tests-hyb-ssb
	tests-output
	tests-plan-parsing
	tests-threads
	)
	add_executable(unit-${target} ${target}.cpp test-utils.cpp)
	set_target_properties(unit-${target} PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
	target_link_libraries(unit-${target} ${gtest_libs})
	install(TARGETS unit-${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

# One test to rule them all
file(GLOB gtest_src *.cpp *.cu)
add_executable(unit-tests ${gtest_src})
set_target_properties(unit-tests PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
target_link_libraries(unit-tests ${gtest_libs})
install(TARGETS unit-tests
	RUNTIME DESTINATION raw
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

enable_testing()
add_test(UT unit-tests)
