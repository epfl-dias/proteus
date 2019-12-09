# Pure, regular C++ files


set(htap_bench_cxx
		htap-cli-flags.cpp
		queries/ch/ch-queries.cpp
		queries/olap-sequence.cpp
    )

#find_package(Threads REQUIRED)



foreach(target
		micro-hybrid
		micro-local
		micro-remote
		server-circular-master
		#server-fork
	)
	add_executable(htap-${target} ${target}.cpp ${htap_bench_cxx})
	target_link_libraries(htap-${target}
			aeolus::bench
			aeolus::plugin
			codegen::olap
			htap::engine
			gflags::gflags
			)
	target_include_directories(htap-${target}
		PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
			$<INSTALL_INTERFACE:include>
			./../../../core/olap
			./../oltp/bench
		PRIVATE
			src)
	install(TARGETS htap-${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)
