set(gtest_libs ${GTEST_MAIN} ${GTEST} proteus::olap oltp::engine)

set(gtest_src
        #tests-snapshotting
        tests-interval-map.cpp
        tests-atomic-bit-set.cpp
        )

foreach (target ${gtest_src})
    get_filename_component(target_name ${target} NAME_WE)
    add_executable(unit-${target_name} ${target} test-utils.cpp)
    target_link_libraries(unit-${target_name} ${gtest_libs})
    install(TARGETS unit-${target_name}
            RUNTIME DESTINATION pelago
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
            )
    add_test(UT unit-${target_name})
endforeach (target)

# One test to rule them all
add_executable(unit-tests-oltp-engine ${gtest_src} test-utils.cpp)
target_link_libraries(unit-tests-oltp-engine ${gtest_libs})
install(TARGETS unit-tests-oltp-engine
        RUNTIME DESTINATION pelago
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

enable_testing()
add_test(UT unit-tests-oltp-engine)
