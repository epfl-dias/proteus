set(gtest_libs ${GTEST_MAIN} ${GTEST} oltp::engine oltp::bench oltp::adaptor)

set(gtest_src
        tests-ycsb
        tests-tpcc
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
add_executable(unit-tests-oltp ${gtest_src} test-utils.cpp)
target_link_libraries(unit-tests-oltp ${gtest_libs})
install(TARGETS unit-tests-oltp
        RUNTIME DESTINATION pelago
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

enable_testing()
add_test(UT unit-tests-oltp)
