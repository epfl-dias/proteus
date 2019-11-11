# Pure, regular C++ files
set(txn_bench_lib_cxx
    #tpcc.cpp
    tpcc_64.cpp
#    micro_ssb.cpp
    )


#set(bench_libs ${bench_libs} -lnuma )
#set(bench_libs ${bench_libs} gflags::gflags ${GLOG} gflags::gflags)
#-----------------------------------------------------------------------------
include_directories(
	"${CMAKE_CURRENT_SOURCE_DIR}"
	SYSTEM "${PROJECT_SOURCE_DIR}/engines/oltp/engine"
	SYSTEM "${PROJECT_SOURCE_DIR}/engines/oltp/lib"
	SYSTEM "${PROJECT_SOURCE_DIR}/codegen"
	)

message("HEREE I AM")
add_library(aeolus_bench SHARED
        ${txn_bench_lib_cxx}
    )

target_link_libraries(aeolus_bench ${default_libs} aeolus_engine)
install(TARGETS aeolus_bench
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )

