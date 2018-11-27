# One test to rule them all
file(GLOB spam_all_vldb *.cpp)
add_executable(spam-workload-vldb ${spam_all_vldb})
set_target_properties(spam-workload-vldb PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
target_link_libraries(spam-workload-vldb ${default_libs})
install(TARGETS spam-workload-vldb
	RUNTIME DESTINATION raw
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
