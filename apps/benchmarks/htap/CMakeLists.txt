foreach(target
		ch-bench
		ch-export-data
	)
	project(htap-${target} VERSION 0.1 LANGUAGES CXX)

	add_executable(htap-${target}
		${target}.cpp
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

	target_link_libraries(htap-${target}
		oltp::bench
		oltp::adaptor
		proteus::olap
		htap::engine
		gflags::gflags
		codegen::olap_flags
		#queries::ch100w
	)

	install(DIRECTORY "queries/clotho"
		DESTINATION ${CMAKE_INSTALL_BINDIR}/benchmarks/htap/queries
		#COMPONENT htap-${target}_${PROTEUS_CPACK_COMP_SUFFIX_BENCHMARKS}
		COMPONENT ${PROTEUS_CPACK_COMP_BENCHMARKS}
	)

	include(proteus-benchmark)
endforeach(target)
