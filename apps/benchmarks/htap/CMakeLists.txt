# Pure, regular C++ files


set(htap_bench_cxx
		htap-cli-flags.cpp
		queries/ch/q01.cpp
		queries/ch/q02.cpp
		queries/ch/q04.cpp
		queries/ch/q06.cpp
		queries/ch/q19.cpp
		queries/ch/ch-queries.cpp
		queries/olap-sequence.cpp
		queries/ch/data-export/data-export-ch.cpp
    )

#find_package(Threads REQUIRED)

install(DIRECTORY "queries/clotho" DESTINATION ${CMAKE_INSTALL_PREFIX}/pelago/benchmarks/htap/queries)

foreach(target
		ch-bench
		ch-export-data
	)
	add_executable(htap-${target} ${target}.cpp ${htap_bench_cxx})
	target_link_libraries(htap-${target}
			oltp::bench
			oltp::adaptor
			proteus::olap
			htap::engine
			gflags::gflags
			codegen::olap_flags
			#queries::ch100w
			)
	target_include_directories(htap-${target}
		PUBLIC
			$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
			$<INSTALL_INTERFACE:include>
		PRIVATE
			src)
	install(TARGETS htap-${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)
