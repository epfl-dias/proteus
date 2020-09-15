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

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap("Template", &argc, &argv);

  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  // init OLTP
  //  OLTP oltp_engine;
  //  oltp_engine.init();

  // txn-variables
  auto num_fields = 2;
  size_t record[] = {1, 2};
  auto intial_records = 100;
  size_t xid = 1;

  // CREATE TABLE T(a INTEGER, b INTEGER)
  storage::ColumnDef columns;
  for (int i = 0; i < num_fields; i++) {
    columns.emplace_back("col_" + std::to_string(i + 1), storage::INTEGER,
                         sizeof(uint64_t));
  }

  auto tbl = storage::Schema::getInstance().create_table(
      "table_one", storage::COLUMN_STORE, columns, intial_records * 10);

  // INSERT INTO T VALUES (5, 10);
  for (auto i = 0; i < intial_records; i++) {
    record[0] = i;
    record[1] = i;
    void *hash_idx = tbl->insertRecord(record, 0, 0, 0);
    tbl->p_index->insert(i, hash_idx);
  }

  // SELECT * FROM T WHERE a=5;

  global_conf::IndexVal *hash_ptr =
      (global_conf::IndexVal *)tbl->p_index->find(5);
  assert(hash_ptr != nullptr);

  hash_ptr->latch.acquire();
  if (txn::CC_MV2PL::is_readable(hash_ptr->t_min, xid)) {
    tbl->getRecordByKey(hash_ptr->VID, nullptr, 0, record);
  } else {
    void *v = hash_ptr->delta_ver->get_readable_ver(xid);
    assert(false && "i dont remember the format of delta return :P");
  }
  hash_ptr->latch.release();

  // UPDATE T SET b = 15 WHERE a=5;

  hash_ptr = (global_conf::IndexVal *)tbl->p_index->find(5);
  if (hash_ptr->write_lck.try_lock()) {
    hash_ptr->latch.acquire();
    ushort col_update_idx = 1;
    tbl->updateRecord(hash_ptr, &(record[col_update_idx]), 0, 0,
                      &col_update_idx, 1);
    hash_ptr->t_min = xid;
    hash_ptr->write_lck.unlock();
    hash_ptr->latch.release();
  } else {
    assert(false && "cannot acquire lock on record");
  }

  return 0;
}
