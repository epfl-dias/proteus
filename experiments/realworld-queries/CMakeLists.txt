foreach(target
	)
	add_executable(${target} ${target}.cpp)
	set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
	target_link_libraries(${target} ${default_libs})
	install(TARGETS ${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)

# One test to rule them all
file(GLOB spam_workload *.cpp)
add_executable(spam-workload ${spam_workload})
set_target_properties(spam-workload PROPERTIES COMPILE_FLAGS ${LLVM_CXXFLAGS})
target_link_libraries(spam-workload ${default_libs})
install(TARGETS spam-workload
	RUNTIME DESTINATION raw
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	)
