# Pure, regular C++ files
set(txn_engine_cxx
    scheduler/worker.cpp
    scheduler/topology.cpp
    scheduler/affinity_manager.cpp
    scheduler/comm_manager.cpp
    storage/memory_manager.cpp
    storage/table.cpp
    #storage/row_store.cpp
    storage/delta_storage.cpp
    transactions/transaction_manager.cpp
    transactions/cc.cpp
    )



#-----------------------------------------------------------------------------

add_library(aeolus_engine SHARED
        ${txn_engine_cxx}
    )
target_link_libraries(aeolus_engine ${LLVM_LIBS} -lnuma)
install(TARGETS aeolus_engine
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )


#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?


