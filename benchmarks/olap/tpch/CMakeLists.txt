foreach(target
	tpch-csv-projections
	tpch-csv-projections-cached
	tpch-csv-selections
	tpch-csv-selections-cached
	tpch-csv-joins
	tpch-csv-groups
	tpch-json-projections
	tpch-json-projections-cached
	tpch-json-selections
	tpch-json-selections-cached
	tpch-json-groups
	tpch-json-groups-cached
	tpch-bin-projections
	tpch-bin-selections
	tpch-bin-joins
	tpch-bin-joins-profiling
	tpch-bin-joins-cached
	tpch-bin-groups
	tpch-bin-groups-profiling
	tpch-bin-groups-cached
	tpch-materializers
	tpch-json-joins
	tpch-json-unnests
	)
	add_executable(${target} ${target}.cpp)
	target_link_libraries(${target} codegen::olap)
	install(TARGETS ${target}
		RUNTIME DESTINATION pelago
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endforeach(target)
