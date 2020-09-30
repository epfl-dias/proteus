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

	# FIXME: remove after old warnings are fixed
	target_compile_options(
				oltp-${target}
			PRIVATE # Not PUBLIC to at least avoid leaking the issue across libs
				"-Wno-error=suggest-override"
				"-Wno-error=suggest-destructor-override"
				"-Wno-missing-variable-declarations"
	)
endforeach(target)




