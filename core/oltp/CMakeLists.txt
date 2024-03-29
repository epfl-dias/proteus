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
  lib/execution/worker-pool.cpp
  lib/execution/worker-schedule-policy.cpp

  #-------- Index
  lib/index/index.cpp
  lib/index/hash-array.cpp
  lib/index/hash-cuckoo-partitioned.cpp
  lib/index/ART/art_node.cpp
  lib/index/ART/art-allocator.cpp

  #-------- Snapshot
  lib/snapshot/snapshot-manager.cpp

  #-------- Storage
  lib/storage/schema.cpp
  lib/storage/schema-info.cpp
  lib/storage/table.cpp
  lib/storage/layout/column-store/column_store.cpp
  lib/storage/layout/column-store/circular-master-column.cpp
  lib/storage/layout/column-store/lazy-column.cpp
  lib/storage/layout/column-store/column.cpp
  lib/storage/multi-version/delta_storage.cpp
  lib/storage/multi-version/mv.cpp
  lib/storage/multi-version/mv-record-list.cpp
  lib/storage/multi-version/mv-attribute-list.cpp
  lib/storage/memory-pool.cpp

  #-------- Transaction
  lib/transactions/transaction_manager.cpp
  lib/transactions/transaction.cpp
  lib/transactions/txn-executor.cpp
  lib/transactions/txn-queue.cpp
  lib/transactions/txn-table.cpp
  #lib/transactions/concurrency-control/cc.cpp
  )


foreach (target ${txn_engine_cxx})
  set_source_files_properties(${target} PROPERTIES COMPILE_FLAGS ${CUDA_CXXFLAGS})
endforeach ()

add_library(${PROJECT_NAME} SHARED
  ${oltp_cxx}
  )

# FIXME: better fix the issue instead of hiding it by disabling the warnings
target_compile_options(${PROJECT_NAME} BEFORE PRIVATE
  -Wno-error=weak-template-vtables
  -Wno-error=unreachable-code
  -Wno-error=missing-noreturn
  -Wno-address-of-packed-member
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
