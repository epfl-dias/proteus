# Pure, regular C++ files


add_subdirectory(bench)
set(txn_bench_cxx
	bench-cli-flags.cpp
	${PROJECT_SOURCE_DIR}/engines/oltp/engine/oltp-cli-flags.cpp
    )







foreach(target
	bench-runner
	)
	add_executable(${target} ${target}.cpp ${txn_bench_cxx})
	target_link_libraries(${target} aeolus_engine aeolus_bench)
	install(TARGETS ${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)




