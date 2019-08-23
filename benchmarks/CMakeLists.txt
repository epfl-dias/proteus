# Pure, regular C++ files
set(txn_bench_cxx
    tpcc.cpp
    micro_ssb.cpp
    )



#-----------------------------------------------------------------------------

add_library(aeolus_bench SHARED
        ${txn_bench_cxx}
    )
target_link_libraries(aeolus_bench ${LLVM_LIBS} -lnuma)
install(TARGETS aeolus_bench
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )


#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?


