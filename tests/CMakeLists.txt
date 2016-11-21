foreach(target
	tests-joins
	tests-json
	tests-output
	tests-plan-parsing
	)
	add_executable(unit-${target} ${target}.cpp)
	set_target_properties(unit-${target} PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
	target_link_libraries(unit-${target} ${gtest_libs})
endforeach(target)

# One test to rule them all
file(GLOB gtest_src *.cpp)
add_executable(unit-tests ${gtest_src})
set_target_properties(unit-tests PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(unit-tests ${gtest_libs})

enable_testing()
add_test(UT unit-tests)
