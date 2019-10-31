set(gtest_libs ${GTEST_MAIN} ${GTEST} ${default_libs} codegen)


#-----------------------------------------------------------------------
# Intel VTune profile support
find_library(VTUNE ittnotify HINTS "/opt/intel/vtune_amplifier/lib64")
if(VTUNE AND VTUNE_ENABLE)
    include_directories(
        SYSTEM "/opt/intel/vtune_amplifier/include"
        )
    set(gtest_libs ${gtest_libs} ${VTUNE} -ldl)
endif()

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
add_executable(unit-tests-oltp ${gtest_src} test-utils.cpp)
target_link_libraries(unit-tests-oltp ${gtest_libs})
install(TARGETS unit-tests-oltp
    RUNTIME DESTINATION pelago
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

enable_testing()
add_test(UT unit-tests-oltp)
