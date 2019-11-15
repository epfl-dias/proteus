# Pure, regular C++ files


add_subdirectory(bench)
set(txn_bench_cxx
	bench-cli-flags.cpp
	${PROJECT_SOURCE_DIR}/engines/oltp/engine/oltp-cli-flags.cpp
    )

find_package(Threads REQUIRED)

foreach(target
	bench-runner
	)
	add_executable(oltp-${target} ${target}.cpp ${txn_bench_cxx})
	target_link_libraries(oltp-${target} aeolus_engine aeolus_bench gflags::gflags
			Threads::Threads -lc++abi)
	install(TARGETS oltp-${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)




