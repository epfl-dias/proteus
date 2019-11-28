set(gtest_libs ${GTEST_MAIN} ${GTEST} codegen::olap)

set(gtest_src
	tests-joins
	tests-json
	tests-cow
	tests-cpu-ssb
	tests-gpu-ssb
	tests-hyb-ssb
	tests-output
	tests-plan-parsing
	tests-threads
	)

foreach(target ${gtest_src})
	get_filename_component(target_name ${target} NAME_WE)
	add_executable(unit-${target_name} ${target} test-utils.cpp)
	target_link_libraries(unit-${target_name} ${gtest_libs})
	install(TARGETS unit-${target_name}
		ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
		LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
		RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
		)
endforeach(target)

# One test to rule them all
add_executable(unit-tests ${gtest_src} test-utils.cpp)
target_link_libraries(unit-tests ${gtest_libs})
install(TARGETS unit-tests
	RUNTIME DESTINATION pelago
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)

enable_testing()
add_test(UT unit-tests)
