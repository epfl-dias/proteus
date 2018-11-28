foreach(target
	symantec-json
	)
	add_executable(${target} ${target}.cpp)
	target_link_libraries(${target} ${default_libs})
	install(TARGETS ${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)
