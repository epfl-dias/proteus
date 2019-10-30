<<<<<<< HEAD
set(gtest_libs ${GTEST_MAIN} ${GTEST} ${default_libs})

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
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
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
=======
set(gtest_libs ${GTEST_MAIN} ${GTEST} ${default_libs} codegen)

set(gtest_src
    #tests-snapshotting
    tests-atomic-bit-set
    )

foreach(target ${gtest_src})
    get_filename_component(target_name ${target} NAME_WE)
    add_executable(unit-${target_name} ${target} test-utils.cpp)
    target_link_libraries(unit-${target_name} ${gtest_libs})
    install(TARGETS unit-${target_name}
        RUNTIME DESTINATION pelago
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )
    add_test(UT unit-${target_name})
endforeach(target)

# One test to rule them all
add_executable(unit-tests-aeolus ${gtest_src} test-utils.cpp)
target_link_libraries(unit-tests-aeolus ${gtest_libs})
install(TARGETS unit-tests-aeolus
    RUNTIME DESTINATION pelago
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

enable_testing()
add_test(UT unit-tests-aeolus)
>>>>>>> f3d085313c1b6800d391ac66580a09c2994c4644
