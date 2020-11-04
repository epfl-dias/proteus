if(FALSE)
# One test to rule them all
set(TS_NAME unit-tests-oltp)

add_executable(${TS_NAME}
	tests-ycsb.cpp
	tests-tpcc.cpp
	test-utils.cpp
)

target_link_libraries(${TS_NAME}
	oltp::engine
	oltp::bench
	oltp::adaptor
)

else()

set(TS_TESTS
	tests-ycsb
	tests-tpcc
)

set(TS_COMMON_SRCS
	test-utils.cpp
)

set(TS_COMMON_LIBS
	oltp::engine
	oltp::bench
	oltp::adaptor
)
endif()

include(proteus-test)
