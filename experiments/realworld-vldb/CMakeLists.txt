###################
###VLDB VERSIONS###
###################

foreach(target
	#spam-bin
	#spam-csv-cached-columnar
	#spam-bin-csv
	)
	add_executable(${target}-vldb ${target}.cpp)
	set_target_properties(${target}-vldb PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
	target_link_libraries(${target}-vldb ${default_libs})
	install(TARGETS ${target}
		RUNTIME DESTINATION raw/
		LIBRARY DESTINATION /lib
		ARCHIVE DESTINATION /lib
		)
endforeach(target)

# One test to rule them all
file(GLOB spam_all_vldb *.cpp)
add_executable(spam-workload-vldb ${spam_all_vldb})
set_target_properties(spam-workload-vldb PROPERTIES COMPILE_FLAGS ${LLVM_CPPFLAGS})
target_link_libraries(spam-workload-vldb ${default_libs})
install(TARGETS spam-workload-vldb
	RUNTIME DESTINATION raw/
	LIBRARY DESTINATION /lib
	ARCHIVE DESTINATION /lib
	)
