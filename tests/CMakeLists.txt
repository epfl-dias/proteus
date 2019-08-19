set(gtest_libs ${GTEST_MAIN} ${GTEST} ${default_libs})

set(gtest_src
    tests-snapshotting
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
add_executable(unit-tests-htap ${gtest_src} test-utils.cpp)
target_link_libraries(unit-tests-htap ${gtest_libs})
install(TARGETS unit-tests-htap
    RUNTIME DESTINATION pelago
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

enable_testing()
add_test(UT unit-tests-htap)
