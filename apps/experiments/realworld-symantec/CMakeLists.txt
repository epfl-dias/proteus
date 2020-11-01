foreach(target
	symantec-json
	)
	add_executable(${target} ${target}.cpp)
	target_link_libraries(${target} proteus::olap)
	install(TARGETS ${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)
