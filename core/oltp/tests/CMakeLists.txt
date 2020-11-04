if(FALSE)
set(TS_NAME unit-tests-oltp-engine)

add_executable(${TS_NAME}
	#tests-snapshotting.cpp
	test-utils.cpp
)

target_link_libraries(${TS_NAME}
	proteus::olap
	oltp::engine
)

else()

set(TS_TESTS
	#tests-snapshotting
)

set(TS_COMMON_SRCS
	test-utils.cpp
)

set(TS_COMMON_LIBS
	proteus::olap
	oltp::engine
)
endif()

include(proteus-test)
