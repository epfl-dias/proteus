foreach(target
	symantec-json
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
