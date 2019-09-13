# Pure, regular C++ files
set(txn_engine_cxx
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
    )


set(engine_libs ${engine_libs} -lnuma )
set(engine_libs ${engine_libs} gflags::gflags ${GLOG} gflags::gflags)

foreach(target ${txn_engine_cxx})
    set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach()

#-----------------------------------------------------------------------------

# grep -c ^processor /proc/cpuinfo 
#execute_process(
#    COMMAND bash "-c" "echo -n hello | sed 's/hello/world/;'" 
#    OUTPUT_VARIABLE FOO
#)

add_library(aeolus_engine SHARED
        ${txn_engine_cxx}
    )
target_link_libraries(aeolus_engine ${LLVM_LIBS} ${engine_libs})
install(TARGETS aeolus_engine
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )


#enable_testing()
#add_test(UT codegen-tests) # Add codegen-specific tests?


