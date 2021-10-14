/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#include "oltp/common/numa-partition-policy.hpp"
#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/schema.hpp"
#include "oltp/storage/storage-utils.hpp"
#include "oltp/transaction/stored-procedure.hpp"
#include "oltp/transaction/transaction_manager.hpp"

namespace storage {

// NOTE: this is meta-table to store information about tables and the number of
// records within each table, however, currently unused as it converts each
// insert record into an update transaction, which causes bottleneck if
// concurrent inserts are targeting the same table.

void Schema::createInformationSchema() {
  // Create the information_schema table
  storage::TableDef columns;
  columns.emplace_back("table_id", storage::INTEGER, sizeof(uint32_t));

  for (auto i = 0; i < g_num_partitions; i++) {
    columns.emplace_back("num_records_part_" + std::to_string(i),
                         storage::INTEGER, sizeof(size_t));
  }

  this->infoSchema = std::allocate_shared<ColumnStore>(
      proteus::memory::PinnedMemoryAllocator<ColumnStore>(), (this->num_tables),
      "information_schema", columns, true, false, 1024,
      storage::NUMAPartitionPolicy::getInstance().getDefaultPartition());

  this->total_mem_reserved += infoSchema->total_memory_reserved;
}

void Schema::insertInfoSchema(table_id_t table_id) {
  assert(infoSchema);
  struct infoSchemaRecord record = {};
  record.table_id = table_id;
  for (auto& i : record.num_records_part) i = 0;

  auto insTx = txn::StoredProcedure(
      [&](txn::TransactionExecutor& executor, txn::Txn& txn, void* params) {
        void* recordPtr =
            infoSchema->insertRecord(&record, txn.txnTs.txn_start_time, 0, 0);
        infoSchema->p_index->insert(table_id, recordPtr);
        return true;
      });

  auto& txnManager = txn::TransactionManager::getInstance();

  auto txnTables = txnManager.getTxnTables();

  if (__likely(txnTables.empty())) {
    txn::TransactionExecutor exec;
    auto txn = txn::Txn::getTxn(
        0, storage::NUMAPartitionPolicy::getInstance().getDefaultPartition());
    insTx.tx(exec, txn, nullptr);
  } else {
    // txnManager is active, so do a proper txn.
    txnManager.executeFullQueue(
        *(txnTables.at(0)), insTx, 0,
        storage::NUMAPartitionPolicy::getInstance().getDefaultPartition());
  }
}

}  // namespace storage
