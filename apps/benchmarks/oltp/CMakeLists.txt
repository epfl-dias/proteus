# Pure, regular C++ files


add_subdirectory(bench)
set(txn_bench_cxx
		bench-cli-flags.cpp
    )

find_package(Threads REQUIRED)

foreach(target
	bench-runner
	)
	add_executable(oltp-${target} ${target}.cpp ${txn_bench_cxx})
	target_link_libraries(oltp-${target}
			aeolus::engine
			aeolus::bench
			gflags::gflags
			codegen::olap_flags
			Threads::Threads
			-lc++abi)
	install(TARGETS oltp-${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)




