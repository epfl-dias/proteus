if(FALSE)
# Unused for now, as there is an assert which prevents to run a single
# test from a compound binary.

# One test to rule them all

# Parameters:
# - TS_NAME Target name of the test suite.

target_link_libraries(${TS_NAME}
	PUBLIC
	${GTEST_MAIN}
	${GTEST}
	)
target_enable_default_warnings(${TS_NAME})

set(_proteus_install_target ${TS_NAME})

#set(_proteus_install_dev ${PROJECT_NAME}_Development)
#set(_proteus_install_bin ${PROJECT_NAME}_Tests)
#set(_proteus_install_dev ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_DEV})
#set(_proteus_install_bin ${PROJECT_NAME}_${PROTEUS_CPACK_COMP_SUFFIX_TESTS})

set(_proteus_install_dev ${PROTEUS_CPACK_COMP_DEV})
set(_proteus_install_bin ${PROTEUS_CPACK_COMP_TESTS})
include(_proteus-install)

enable_testing()
add_test(NAME UT
	COMMAND ${TS_NAME}
	WORKING_DIRECTORY ${CMAKE_INSTALL_BINDIR}
)

else()
# Slower builds as we link a binary per unit-test. More "Magical variable"
# as well. Hopefully one day the assert will be removed, or put in such
# a way we can run a single unit test from a test suite-wide executable.

# Parameters:
# - TS_TESTS Target name of the unit tests to build.
# - TS_COMMON_SRCS Additional source files shared among all unit tests.
# - TS_COMMON_LIBS Additional libraries shared among all unit tests.
foreach(target ${TS_TESTS})
	add_proteus_unit_test_nolib(unit-${target} ${target}.cpp ${TS_COMMON_SRCS})
	proteus_target_link_test_runner(unit-${target})

	target_link_libraries(unit-${target} PUBLIC ${TS_COMMON_LIBS})

	# Not sure we need to do it for each target, we might be able to
	# do it only once.
	enable_testing()
endforeach(target)

endif()
