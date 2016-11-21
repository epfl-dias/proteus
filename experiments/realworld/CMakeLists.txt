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
	set_target_properties(${target} PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
	target_link_libraries(${target} ${default_libs})
endforeach(target)

# One test to rule them all
