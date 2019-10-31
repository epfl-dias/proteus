# Pure, regular C++ files
set(txn_engine_cxx
    glo.cpp
    scheduler/worker.cpp
    scheduler/topology.cpp
    scheduler/affinity_manager.cpp
    scheduler/comm_manager.cpp
    storage/memory_manager.cpp
    storage/table.cpp
    storage/row_store.cpp
    storage/column_store.cpp
    storage/delta_storage.cpp
    transactions/transaction_manager.cpp
    #transactions/cc.cpp
    snapshot/cor_arena.cpp
    snapshot/cow_arena.cpp
    snapshot/cor_const_arena.cpp
    indexes/hash_array.cpp
    oltp-cli-flags.cpp
    )

add_library(aeolus_engine SHARED
        ${txn_engine_cxx}
    )

target_link_libraries(aeolus_engine)
install(TARGETS aeolus_engine
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )
