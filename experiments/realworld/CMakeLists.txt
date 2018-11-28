foreach(target
	spam-bin
	spam-csv
	spam-json
	spam-bin-csv
	spam-bin-json
	spam-csv-json
	spam-bin-csv-json
	spam-bin-csv-json-v2
	spam-csv-cached
	spam-csv-cached-columnar
	spam-json-cached
	)
	add_executable(${target} ${target}.cpp)
	target_link_libraries(${target} ${default_libs})
	install(TARGETS ${target}
		RUNTIME DESTINATION raw
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)
