project(oltp VERSION 0.1 LANGUAGES CXX)

add_subdirectory(adaptor)

# Unit-tests
if (UNIT_TESTS)
    add_subdirectory(tests)
endif ()

set(oltp_cxx

        #-------- Common
        lib/common/common.cpp
        lib/common/constants.cpp
        lib/common/oltp-cli-flags.cpp
        lib/common/numa-partition-policy.cpp
        lib/common/utils.cpp

        #-------- Execution
        lib/execution/worker.cpp

        #-------- Index
        lib/index/hash_array.cpp
        # lib/index/ART/adaptive_radix_tree_index.cpp

        #-------- Snapshot
        lib/snapshot/snapshot-manager.cpp
        lib/snapshot/cor_arena.cpp
        lib/snapshot/cow_arena.cpp
        lib/snapshot/cor_const_arena.cpp

        #-------- Storage
        lib/storage/schema.cpp
        lib/storage/table.cpp
        lib/storage/layout/column-store/column_store.cpp
        lib/storage/layout/column-store/circular-master-column.cpp
        lib/storage/layout/column-store/lazy-column.cpp
        lib/storage/layout/column-store/column.cpp
        # lib/storage/layout/row_store.cpp
        lib/storage/multi-version/delta_storage.cpp
        lib/storage/multi-version/mv.cpp
        lib/storage/multi-version/mv-record-list.cpp
        lib/storage/multi-version/mv-attribute-list.cpp

        #-------- Transaction
        lib/transactions/transaction_manager.cpp
        lib/transactions/transaction.cpp
        #lib/transactions/concurrency-control/cc.cpp
        )


foreach (target ${txn_engine_cxx})
    set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach ()

add_library(${PROJECT_NAME} SHARED
  ${oltp_cxx}
  )

# If we have compiler requirements for this library, list them
# here
target_compile_features(${PROJECT_NAME}
  PUBLIC
  cxx_std_20
  )

# Depend on a library that we defined in the top-level file
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  cuckoo::cuckoo
  proteus::olap
  cli-flags
  )

include(proteus-lib)
add_library(oltp::engine ALIAS ${PROJECT_NAME})
