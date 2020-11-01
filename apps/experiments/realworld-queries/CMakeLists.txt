# One test to rule them all
file(GLOB spam_workload *.cpp)
add_executable(spam-workload ${spam_workload})
target_link_libraries(spam-workload proteus::olap)
install(TARGETS spam-workload
	RUNTIME DESTINATION pelago
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
