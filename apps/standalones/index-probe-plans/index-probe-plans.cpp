/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <cli-flags.hpp>
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>

#include "oltp.hpp"

storage::Table *tbl = nullptr;
auto num_fields = 2;
size_t record[] = {1, 2};
auto intial_records = 100;

bool insert_query(uint64_t xid, ushort master_ver, ushort delta_ver,
                  ushort partition_id) {
  // INSERT INTO T VALUES (5, 10);

  for (auto i = 0; i < intial_records; i++) {
    record[0] = i;
    record[1] = i;
    void *hash_idx = tbl->insertRecord(record, xid, partition_id, master_ver);
    tbl->p_index->insert(i, hash_idx);

    LOG(INFO) << "INSERTED: " << i;
  }
  LOG(INFO) << "INSERT QUERY SUCCESS";
  return true;
}

bool update_query(uint64_t xid, ushort master_ver, ushort delta_ver,
                  ushort partition_id) {
  // UPDATE T SET b = 15 WHERE a=5;

  auto *hash_ptr = (global_conf::IndexVal *)tbl->p_index->find(5);
  if (hash_ptr->write_lck.try_lock()) {
    hash_ptr->latch.acquire();
    ushort col_update_idx = 1;
    record[col_update_idx] = 15;
    tbl->updateRecord(xid, hash_ptr, &(record[col_update_idx]), master_ver,
                      delta_ver, &col_update_idx, 1);
    hash_ptr->latch.release();
    hash_ptr->write_lck.unlock();
    LOG(INFO) << "UPDATE QUERY SUCCESS";
    return true;
  } else {
    LOG(INFO) << "UPDATE QUERY FAILED";
    return false;
  }
}

bool select_query(uint64_t xid, ushort master_ver, ushort delta_ver,
                  ushort partition_id) {
  // SELECT * FROM T WHERE a=5;

  auto *hash_ptr = (global_conf::IndexVal *)tbl->p_index->find(5);

  if (hash_ptr != nullptr) {
    hash_ptr->latch.acquire();
    tbl->getRecordByKey(hash_ptr->VID, nullptr, 0, &(record[0]));
    hash_ptr->latch.release();
    LOG(INFO) << "SELECT VALUE GOT: [0]: " << record[0];
    LOG(INFO) << "SELECT VALUE GOT: [1]: " << record[1];

    return true;
  } else {
    assert(hash_ptr != nullptr && "Key not found.");
    return false;
  }
}

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap("Template", &argc, &argv);

  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  // init OLTP
  OLTP oltp_engine;
  oltp_engine.init();

  // CREATE TABLE T(a INTEGER, b INTEGER)
  storage::ColumnDef columns;
  for (int i = 0; i < num_fields; i++) {
    columns.emplace_back("col_" + std::to_string(i + 1), storage::INTEGER,
                         sizeof(uint64_t));
  }

  tbl = storage::Schema::getInstance().create_table(
      "table_one", storage::COLUMN_STORE, columns, intial_records * 10);

  // Insert multiple values.
  // INSERT INTO T VALUES (5, 10);
  oltp_engine.enqueue_query(insert_query);

  // SELECT * FROM T WHERE a=5;
  oltp_engine.enqueue_query(select_query);

  // UPDATE T SET b = 15 WHERE a=5;
  oltp_engine.enqueue_query(update_query);

  // SELECT * FROM T WHERE a=5;
  oltp_engine.enqueue_query(select_query);

  return 0;
}
