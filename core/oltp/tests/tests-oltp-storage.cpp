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

#include <gtest/gtest.h>
#include <sys/mman.h>

#include <platform/common/common.hpp>

#include "oltp/storage/layout/column_store.hpp"
#include "oltp/storage/schema.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/concurrency-control/concurrency-control.hpp"
#include "oltp/transaction/transaction.hpp"
#include "test-utils.hpp"

constexpr size_t n_columns = 10;
constexpr size_t n_records = 10;

constexpr size_t txnPartitionID = 0;
constexpr size_t txnMasterVersion = 0;

::testing::Environment *const pools_env =
    ::testing::AddGlobalTestEnvironment(new TestEnvironment);

class OLTPStorageTest : public ::testing::Test {
 protected:
  OLTPStorageTest() { LOG(INFO) << "OLTPStorageTest()"; }
  ~OLTPStorageTest() override { LOG(INFO) << "~OLTPStorageTest()"; }

  //  void SetUp() override;
  //  void TearDown() override;

  const char *testPath = TEST_OUTPUTS "/tests-oltpstorage/";
  const char *catalogJSON = "inputs";

 public:
};

void updateRecord(const std::shared_ptr<storage::Table> &table, txn::Txn &txn,
                  size_t nRec) {
  LOG(INFO) << "Updating tbl: " << table->name;
  for (size_t i = 0; i < nRec; i++) {
    LOG(INFO) << "Updating rec:" << i;
    std::vector<uint64_t> record(n_columns, i + 1);
    auto *recordPtr =
        static_cast<global_conf::IndexVal *>(table->p_index->find(i));
    EXPECT_EQ(recordPtr->VID, i);
    EXPECT_TRUE(recordPtr != nullptr);
    LOG(INFO) << "\t VID: " << recordPtr->VID;
    // update record here

    recordPtr->writeWithLatch(
        [&](global_conf::IndexVal *idx_ptr) {
          LOG(INFO) << "\t VID: " << idx_ptr->VID;
          LOG(INFO) << "\tIDX_PTR:" << reinterpret_cast<uintptr_t>(idx_ptr);
          table->updateRecord(txn, idx_ptr, record.data(), nullptr, -1);
        },
        txn, table->table_id);

    EXPECT_EQ(recordPtr->VID, i);
    std::vector<uint64_t> record2(n_columns, 99999);

    // std::atomic_thread_fence(std::memory_order_seq_cst);

    EXPECT_TRUE(txn::CC_MV2PL::is_readable(*recordPtr, txn.txnTs));

    recordPtr->readWithLatch([&](global_conf::IndexVal *idx_ptr) {
      table->getIndexedRecord(txn.txnTs, *idx_ptr, record2.data(), nullptr, -1);

      for (auto u = 0; u < record.size(); u++) {
        EXPECT_EQ(record[u], record2[u]);
      }

      for (auto &x : record2) {
        EXPECT_EQ(i + 1, x);
      }
    });
  }
}

void insertRecords(const std::shared_ptr<storage::Table> &table, txn::Txn &txn,
                   size_t nRec) {
  for (size_t i = 0; i < nRec; i++) {
    std::vector<uint64_t> record(n_columns, i);
    void *recordPtr =
        table->insertRecord(record.data(), txn.txnTs.txn_start_time,
                            txn.partition_id, txnMasterVersion);
    assert(recordPtr);
    table->p_index->insert(i, recordPtr);
  }
}

void validate_record(const std::shared_ptr<storage::Table> &table,
                     txn::Txn &txn, size_t nRec, xid_t expectedTmin,
                     size_t expectedValOffset) {
  LOG(INFO) << "Validating records for " << table->name
            << " | totalRecords: " << nRec;
  for (size_t i = 0; i < nRec; i++) {
    LOG(INFO) << "\tRecord: " << i;
    auto *recordPtr =
        static_cast<global_conf::IndexVal *>(table->p_index->find(i));

    EXPECT_EQ(recordPtr->VID, i);
    EXPECT_EQ(recordPtr->ts.t_min, expectedTmin);

    LOG(INFO) << "\t\tVID: " << recordPtr->VID;
    LOG(INFO) << "\t\tTmin: " << recordPtr->ts.t_min;
    LOG(INFO) << "\t\tCurrTxn: " << txn.txnTs.txn_start_time;

    std::vector<uint64_t> record(n_columns, 0);
    // txn::TxnTs

    recordPtr->readWithLatch([&](global_conf::IndexVal *idx_ptr) {
      table->getIndexedRecord(txn.txnTs, *idx_ptr, record.data(), nullptr, -1);

      for (auto &x : record) {
        EXPECT_EQ(x, i + expectedValOffset);
      }
    });
  }
}

TEST_F(OLTPStorageTest, ValidateWrite) {
  auto *schema = &storage::Schema::getInstance();

  storage::TableDef columns;
  for (int i = 0; i < n_columns; i++) {
    columns.emplace_back("col_" + std::to_string(i + 1), storage::INTEGER,
                         sizeof(uint64_t));
  }

  auto table1 = schema->create_table(
      "ValidateWrite_tbl_1", storage::COLUMN_STORE, columns, n_records / 3);
  auto table2 = schema->create_table(
      "ValidateWrite_tbl_2", storage::COLUMN_STORE, columns, n_records / 2);

  auto table3 = schema->create_table("ValidateWrite_tbl_3",
                                     storage::COLUMN_STORE, columns, n_records);

  // start a global txn
  auto txn1 = txn::Txn::getTxn(0, txnPartitionID);
  auto expectedTmin = txn1.txnTs.txn_start_time;

  // record insertion

  insertRecords(table1, txn1, n_records / 3);
  insertRecords(table2, txn1, n_records / 2);
  insertRecords(table3, txn1, n_records);

  LOG(INFO) << "InsertValTest";
  // validate
  auto txn = txn::Txn::getTxn(0, txnPartitionID);
  validate_record(table1, txn, n_records / 3, expectedTmin, 0);
  validate_record(table2, txn, n_records / 2, expectedTmin, 0);
  validate_record(table3, txn, n_records, expectedTmin, 0);

  expectedTmin = txn.txnTs.txn_start_time;

  LOG(INFO) << "UpdateTest";
  updateRecord(table1, txn, n_records / 3);
  updateRecord(table2, txn, n_records / 2);
  updateRecord(table3, txn, n_records);
  LOG(INFO) << "ValidateAgain";
  validate_record(table1, txn, n_records / 3, expectedTmin, 1);
  validate_record(table2, txn, n_records / 2, expectedTmin, 1);
  validate_record(table3, txn, n_records, expectedTmin, 1);

  table1.reset();
  table2.reset();
  table3.reset();
}

TEST_F(OLTPStorageTest, ReadFromMV) {
  auto *schema = &storage::Schema::getInstance();
  storage::Schema::getInstance().report();
  storage::TableDef columns;
  for (int i = 0; i < n_columns; i++) {
    columns.emplace_back("col_" + std::to_string(i + 1), storage::INTEGER,
                         sizeof(uint64_t));
  }

  auto table1 = schema->create_table("ReadFromMV_tbl_1", storage::COLUMN_STORE,
                                     columns, n_records / 3);
  auto table2 = schema->create_table("ReadFromMV_tbl_2", storage::COLUMN_STORE,
                                     columns, n_records / 2);

  auto table3 = schema->create_table("ReadFromMV_tbl_3", storage::COLUMN_STORE,
                                     columns, n_records);

  storage::Schema::getInstance().report();
  // start a global txn
  auto txn1 = txn::Txn::getTxn(0, txnPartitionID);
  auto expectedTmin = txn1.txnTs.txn_start_time;

  // record insertion
  insertRecords(table1, txn1, n_records / 3);
  insertRecords(table2, txn1, n_records / 2);
  insertRecords(table3, txn1, n_records);

  // validate
  auto txn2 = txn::Txn::getTxn(0, txnPartitionID);
  validate_record(table1, txn2, n_records / 3, expectedTmin, 0);
  validate_record(table2, txn2, n_records / 2, expectedTmin, 0);
  validate_record(table3, txn2, n_records, expectedTmin, 0);

  updateRecord(table1, txn2, n_records / 3);
  updateRecord(table2, txn2, n_records / 2);
  updateRecord(table3, txn2, n_records);
  LOG(INFO) << "Records updated by tMin: " << txn2.txnTs.txn_start_time;

  // expected Tmin arg is for the Tmin on the indexRecord.
  validate_record(table1, txn1, n_records / 3, txn2.txnTs.txn_start_time, 0);
  validate_record(table2, txn1, n_records / 2, txn2.txnTs.txn_start_time, 0);
  validate_record(table3, txn1, n_records, txn2.txnTs.txn_start_time, 0);

  table1.reset();
  table2.reset();
  table3.reset();
}
