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

#include <aeolus-plugin.hpp>
#include <cli-flags.hpp>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <oltp.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

static storage::Table *tbl;
static auto num_fields = 2;
static size_t record[] = {1, 2};
static auto initial_records = 100;

bool insert_query(uint64_t xid, ushort master_ver, ushort delta_ver,
                  ushort partition_id) {
  // INSERT INTO T VALUES (5, 10);

  for (auto i = 0; i < initial_records; i++) {
    record[0] = i + 2;
    record[1] = i + 1175;
    void *hash_idx = tbl->insertRecord(record, xid, partition_id, master_ver);
    tbl->p_index->insert(i, hash_idx);

    LOG(INFO) << "INSERTED: " << i;
  }
  LOG(INFO) << "INSERT QUERY SUCCESS";
  return true;
}

bool update_query(uint64_t xid, ushort master_ver, ushort delta_ver,
                  ushort partition_id) {
  LOG(INFO) << xid << " " << master_ver << " " << delta_ver << " "
            << partition_id;
  // UPDATE T SET b = 15 WHERE a=5;

  auto *hash_ptr = (global_conf::IndexVal *)tbl->p_index->find(5);
  if (hash_ptr->write_lck.try_lock()) {
    hash_ptr->latch.acquire();
    column_id_t col_update_idx = 1;
    record[col_update_idx] = 15;
    tbl->updateRecord(xid, hash_ptr, &(record[col_update_idx]), delta_ver,
                      &col_update_idx, 1, master_ver);
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
    // tbl->getIndexedRecord(xid, hash_ptr, &(record[0]), nullptr, 0);
    hash_ptr->latch.release();
    LOG(INFO) << "SELECT VALUE GOT: [0]: " << record[0];
    LOG(INFO) << "SELECT VALUE GOT: [1]: " << record[1];

    return true;
  } else {
    assert(hash_ptr != nullptr && "Key not found.");
    return false;
  }
}

struct session {
  uint64_t xid;
  ushort master_ver;
  ushort delta_ver;
  ushort partition_id;
};

extern session TheSession;

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap("Template", &argc, &argv);

  CatalogParser::getInstance();

  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  // init OLTP
  OLTP oltp_engine{};
  oltp_engine.init();

  // CREATE TABLE T(a INTEGER, b INTEGER)
  storage::TableDef columns;
  for (int i = 0; i < num_fields; i++) {
    columns.emplace_back("col_" + std::to_string(i + 1), storage::INTEGER,
                         sizeof(uint64_t));
  }

  tbl = storage::Schema::getInstance().create_table(
      "table_one", storage::COLUMN_STORE, columns, initial_records * 10);

  // FIXME: enqueue_query interface broken due to OLTP interface changes.

  // Insert multiple values.
  // INSERT INTO T VALUES (5, 10);
  // oltp_engine.enqueue_query(insert_query);

  // SELECT * FROM T WHERE a=5;
  // oltp_engine.enqueue_query(select_query);

  // UPDATE T SET b = 15 WHERE a=5;
  //  oltp_engine.enqueue_query(update_query);

  // SELECT * FROM T WHERE a=5;
  // oltp_engine.enqueue_query(select_query);

  RelBuilderFactory factory{__FUNCTION__};

  auto builder =
      factory.getBuilder()
          .scan("table_one<block-remote>", {"col_1", "col_2"},
                CatalogParser::getInstance(), pg{AeolusRemotePlugin::type})
          .unpack()
          .filter([](const auto &arg) -> expression_t {
            return lt(arg["col_1"], int64_t{5});
          })
          //          .project([](const auto &arg) -> std::vector<expression_t>
          //          {
          //            return {(arg["col_2"] + int64_t{1}).as("test",
          //            "test_b")};
          //          })
          .update([](const auto &arg) -> expression_t {
            return expressions::RecordConstruction{
                (arg["col_1"] + int64_t{2000})
                    .as("table_one<block-remote>", "col_2")}
                .as("table_one<block-remote>", "update");
          })
          .print(pg{"pm-csv"})
          .prepare();

  auto builder2 =
      RelBuilderFactory{"f2"}
          .getBuilder()
          .scan("table_one<block-remote>", {"col_1", "col_2"},
                CatalogParser::getInstance(), pg{AeolusRemotePlugin::type})
          .unpack()
          .filter([](const auto &arg) -> expression_t {
            return lt(arg["col_1"], int64_t{10});
          })
          .project([](const auto &arg) -> std::vector<expression_t> {
            return {(arg["col_1"]).as("test", "test_a"),
                    (arg["col_2"]).as("test", "test_b")};
          })
          .print(pg{"pm-csv"}, "final")
          .prepare();

  oltp_engine.snapshot();
  //  oltp_engine.enqueue_query([&](auto xid, auto master_ver, auto delta_ver,
  //                                auto partition_id) -> bool {
  //    TheSession = session{xid, master_ver, delta_ver, partition_id};
  //    LOG(INFO) << builder.execute();
  //    return true;
  //  });

  std::this_thread::sleep_for(std::chrono::seconds{5});

  oltp_engine.snapshot();
  //  oltp_engine.enqueue_query([&](auto xid, auto master_ver, auto delta_ver,
  //                                auto partition_id) -> bool {
  //    TheSession = session{xid, master_ver, delta_ver, partition_id};
  //    LOG(INFO) << builder2.execute();
  //    return true;
  //  });

  std::this_thread::sleep_for(std::chrono::seconds{5});

  return 0;
}
