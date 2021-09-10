/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef BENCH_YCSB_HPP_
#define BENCH_YCSB_HPP_

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <olap/operators/relbuilder-factory.hpp>
#include <oltp/common/constants.hpp>
#include <oltp/interface/bench.hpp>
#include <oltp/storage/table.hpp>
#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/topology.hpp>
#include <thread>
#include <utility>

#include "aeolus-plugin.hpp"
#include "olap/operators/relbuilder.hpp"
#include "olap/plan/catalog-parser.hpp"
#include "zipf.hpp"

namespace bench {

#define THREAD_LOCAL false
#define PARTITION_LOCAL false
#define YCSB_MIXED_OPS 1

/*

        Benchmark: Yahoo! Cloud Serving Benchmark

        Description:

        Tunable Parameters:

          - num_fields
          - num_records
          - zipf_theta
          - num_ops_per_txn
          - write_threshold
          - num_workers ( for zipf )

*/

class YCSB : public Benchmark {
 public:
  YCSB(YCSB &&) = delete;
  YCSB &operator=(YCSB &&) = delete;
  YCSB(const YCSB &) = delete;
  YCSB &operator=(const YCSB &) = delete;

 private:
  const int num_fields;
  const int num_records;
  // const int num_iterations_per_worker;
  const int num_ops_per_txn;
  const double write_threshold;

  bench_utils::ZipfianGenerator<size_t> zipf;

  // uint64_t recs_per_server;
  storage::Schema *schema;
  std::shared_ptr<storage::Table> ycsb_tbl{};

  const uint num_of_col_upd_per_op;
  const uint num_of_col_read_per_op;
  const uint num_col_read_offset_per_op;

  std::vector<column_id_t> col_idx_upd{};
  std::vector<column_id_t> col_idx_read{};

  struct YCSB_TXN_OP {
    uint64_t key;
    txn::OP_TYPE op_type;
    void *rec;
  };  // __attribute__((aligned(64)));

  struct YCSB_TXN {
    struct YCSB_TXN_OP *ops;
    uint n_ops;
    // bool read_only;

    ~YCSB_TXN() { delete ops; }
  };  // __attribute__((aligned(64)));

  auto *get_query_struct(partition_id_t pid) const {
    auto _txn_mem = (struct YCSB_TXN *)MemoryManager::mallocPinnedOnNode(
        sizeof(struct YCSB_TXN), pid);
    assert(_txn_mem);
    assert(num_ops_per_txn > 0);
    _txn_mem->ops = (struct YCSB_TXN_OP *)MemoryManager::mallocPinnedOnNode(
        sizeof(struct YCSB_TXN_OP) * num_ops_per_txn, pid);
    assert(_txn_mem->ops);
    return _txn_mem;
  }
  static void free_query_struct(struct YCSB_TXN *_txn_mem) {
    MemoryManager::freePinned(_txn_mem->ops);
    MemoryManager::freePinned(_txn_mem);
  }

 public:
  void pre_run(worker_id_t wid, xid_t xid, partition_id_t partition_id,
               master_version_t master_ver) {
    assert(xid < TXN_ID_BASE);
    uint64_t to_ins = num_records / num_max_workers;
    uint64_t start = to_ins * wid;

    auto q_ptr = get_query_struct(partition_id);
    txn::TransactionExecutor executorTmp;

    auto *txnPtr = (txn::Txn *)malloc(sizeof(txn::Txn));

    //    xid_t Txn::getTxn(Txn *txnPtr, worker_id_t workerId, partition_id_t
    //    partitionId,
    //                      bool readOnly)

    for (uint64_t i = start; i < (start + to_ins); i++) {
      std::vector<uint64_t> tmp(num_fields, 1);
      gen_insert_txn(i, tmp.data(), q_ptr);
      txn::Txn::getTxn(txnPtr, wid, partition_id);
      this->exec_txn(executorTmp, *txnPtr, q_ptr);
    }
    free(txnPtr);
    free_query_struct(q_ptr);
  }

  static void gen_insert_txn(uint64_t key, void *rec, struct YCSB_TXN *q_ptr) {
    q_ptr->ops[0].op_type = txn::OPTYPE_INSERT;
    // txn.ops[0].data_table = ycsb_tbl;
    q_ptr->ops[0].key = key;
    q_ptr->ops[0].rec = rec;
    q_ptr->n_ops = 1;
  }
  static void gen_upd_txn(uint64_t key, void *rec, struct YCSB_TXN *q_ptr) {
    q_ptr->ops[0].op_type = txn::OPTYPE_UPDATE;
    // txn.ops[0].data_table = ycsb_tbl;
    q_ptr->ops[0].key = key;
    q_ptr->ops[0].rec = rec;
    q_ptr->n_ops = 1;
  }

  void gen_txn(worker_id_t wid, struct YCSB_TXN *txn,
               partition_id_t partition_id) {
    static thread_local auto recs_per_thread =
        this->num_records / this->num_active_workers;
    static thread_local uint64_t rec_key_iter = 0;
    bool is_duplicate = false;
#if YCSB_MIXED_OPS

    static thread_local uint num_w_ops = write_threshold * num_ops_per_txn;

#else

    txn::OP_TYPE op;
    ushort wid_n = wid % num_active_workers;

    if (wid_n >= (write_threshold * (double)num_active_workers)) {
      op = txn::OPTYPE_LOOKUP;
      // txn->read_only = true;
    } else {
      op = txn::OPTYPE_UPDATE;
      // txn->read_only = false;
    }

#endif
    txn->n_ops = num_ops_per_txn;
    for (int i = 0; i < txn->n_ops; i++) {
      // txn->ops[i].data_table = ycsb_tbl;
      txn->ops[i].rec = nullptr;

#if YCSB_MIXED_OPS
      if (i < (txn->n_ops - num_w_ops)) {
        txn->ops[i].op_type = txn::OPTYPE_LOOKUP;
      } else {
        txn->ops[i].op_type = txn::OPTYPE_UPDATE;
      }
#else
      txn->ops[i].op_type = op;

#endif

#if THREAD_LOCAL
      // In a round-robin way, each thread will operate on its own data block,
      // so there will be no conflicts between two workers.

      txn->ops[i].key =
          (wid * recs_per_thread) + ((rec_key_iter++) % recs_per_thread);
#else
      do {
        // make op
        txn->ops[i].key = zipf.nextval(partition_id, wid);
        is_duplicate = false;

        for (int j = 0; j < i; j++) {
          if (txn->ops[i].key == txn->ops[j].key) {
            is_duplicate = true;
            break;
          }
        }
      } while (is_duplicate);
#endif
    }
  }

  bool exec_txn(txn::TransactionExecutor &executor, txn::Txn &txn,
                void *params) {
    auto *txn_stmts = static_cast<struct YCSB_TXN *>(params);
    int n = txn_stmts->n_ops;

    // static thread_local int n = this->num_ops_per_txn;
    static thread_local ushort num_col_upd = this->num_of_col_upd_per_op;
    static thread_local ushort num_col_read = this->num_of_col_read_per_op;
    static thread_local std::vector<column_id_t> col_idx_read_local(
        col_idx_read);
    static thread_local std::vector<column_id_t> col_idx_update_local(
        col_idx_upd);
    static thread_local std::vector<uint64_t> read_loc(num_fields + 2, 0);

    static thread_local std::vector<global_conf::IndexVal *>
        hash_ptrs_lock_acquired(this->num_ops_per_txn, nullptr);
    uint num_locks = 0;

    /* Acquire locks for updates*/

    for (int i = 0; i < n; i++) {
      struct YCSB_TXN_OP op = txn_stmts->ops[i];

      switch (op.op_type) {
        case txn::OPTYPE_UPDATE: {
          auto *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);
          if (hash_ptr->write_lck.try_lock()) {
            hash_ptrs_lock_acquired[num_locks] = hash_ptr;
            num_locks++;
          } else {
            txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired.data(),
                                         num_locks);
            return false;
          }
          break;
        }
        case txn::OPTYPE_LOOKUP:
        case txn::OPTYPE_INSERT:
        default:
          break;
      }
    }

    txn.undoLogMap.reserve(n);
    // perform lookups/ updates / inserts
    for (int i = 0; i < n; i++) {
      struct YCSB_TXN_OP op = txn_stmts->ops[i];
      switch (op.op_type) {
        case txn::OPTYPE_LOOKUP: {
          auto *recordPtr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          recordPtr->readWithLatch([&](global_conf::IndexVal *idx_ptr) {
            ycsb_tbl->getIndexedRecord(txn.txnTs, *idx_ptr, read_loc.data(),
                                       col_idx_read_local.data(), num_col_read);
          });
          break;
        }

        case txn::OPTYPE_UPDATE: {
          auto *recordPtr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          recordPtr->writeWithLatch(
              [&](global_conf::IndexVal *idx_ptr) {
                ycsb_tbl->updateRecord(txn.txnTs.txn_start_time, idx_ptr,
                                       op.rec, txn.delta_version,
                                       col_idx_update_local.data(), num_col_upd,
                                       txn.master_version);
              },
              txn, ycsb_tbl->table_id);
          recordPtr->write_lck.unlock();
          break;
        }
        case txn::OPTYPE_INSERT: {
          void *recordPtr =
              ycsb_tbl->insertRecord(op.rec, txn.txnTs.txn_start_time,
                                     txn.partition_id, txn.master_version);
          ycsb_tbl->p_index->insert(op.key, recordPtr);
          break;
        }
        default:
          break;
      }
    }
    // txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired.data(), num_locks);

    // FIXME: commitTs should be acquire after acquiring all logs, not use the
    // start_time. but then, all versions create need to be updated also.
    txn.commit_ts = txn.txnTs.txn_start_time;
    return true;
  }

  void init() override {}
  void deinit() override {
    ycsb_tbl.reset();
    zipf.~ZipfianGenerator();
  }
  ~YCSB() override = default;

  BenchQueue *getBenchQueue(worker_id_t workerId,
                            partition_id_t partitionId) override {
    return dynamic_cast<BenchQueue *>(
        new YCSBTxnGen(*this, workerId, partitionId));
  }

  // private:
  YCSB(std::string name = "YCSB", int num_fields = 2, int num_records = 1000000,
       double theta = 0.5, int num_iterations_per_worker = 1000000,
       int num_ops_per_txn = 2, double write_threshold = 0.5,
       int num_active_workers = -1, int num_max_workers = -1,
       ushort num_partitions = 1, bool layout_column_store = true,
       uint num_of_col_upd = 1, uint num_of_col_read = 1,
       uint num_col_read_offset = 0)
      : Benchmark(std::move(name), num_active_workers, num_max_workers,
                  num_partitions),
        num_fields(num_fields),
        num_records(num_records),
        // num_iterations_per_worker(num_iterations_per_worker),
        num_ops_per_txn(num_ops_per_txn),
        write_threshold(write_threshold),
        zipf(num_records, theta,
             num_max_workers == -1 ? topology::getInstance().getCoreCount()
                                   : num_max_workers,
             num_partitions, PARTITION_LOCAL, THREAD_LOCAL),
        num_of_col_upd_per_op(num_of_col_upd),
        num_of_col_read_per_op(num_of_col_read),
        num_col_read_offset_per_op(num_col_read_offset) {
    if (num_max_workers == -1)
      num_max_workers = topology::getInstance().getCoreCount();
    if (num_active_workers == -1)
      this->num_active_workers = topology::getInstance().getCoreCount();

    assert(num_of_col_upd_per_op <= num_fields);
    assert(this->num_records % this->num_max_workers == 0 &&
           "Total number of records should be divisible by total # cores");

    if (num_of_col_read_per_op + num_col_read_offset_per_op > num_fields) {
      assert(false && "read-col offset + #_read_col more than total columns");
    }

    for (uint t = 0; t < num_fields; t++) {
      col_idx_upd.emplace_back(t);
    }
    for (uint t = num_col_read_offset_per_op;
         t < (num_col_read_offset_per_op + num_of_col_read_per_op); t++) {
      col_idx_read.emplace_back(t);
    }

    this->schema = &storage::Schema::getInstance();
    LOG(INFO) << "Workers: " << (uint)(this->num_active_workers);
    LOG(INFO) << "Max-Workers: " << (uint)(this->num_max_workers);
    this->YCSB::init();

    storage::TableDef columns;
    for (int i = 0; i < num_fields; i++) {
      columns.emplace_back("col_" + std::to_string(i + 1), storage::INTEGER,
                           sizeof(uint64_t));
    }
    auto num_record_capacity = num_records;
    auto rec_per_worker = num_records / num_max_workers;
    auto worker_per_partition =
        topology::getInstance().getCpuNumaNodes()[0].local_cores.size();
    if (num_max_workers % worker_per_partition != 0 && num_partitions > 1) {
      num_record_capacity =
          rec_per_worker * worker_per_partition * num_partitions;
    }

    ycsb_tbl = schema->create_table(
        "ycsb_tbl",
        (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
        columns, num_record_capacity);
  }

 private:
  static PreparedStatement export_tbl(bool output_binary,
                                      std::string output = "ycsb_tbl") {
    constexpr static auto csv_plugin = "pm-csv";
    constexpr static auto bin_plugin = "block";
    static RelBuilderFactory ctx{"YCSBDataExporter_CH"};

    auto rel =
        ctx.getBuilder()
            .scan("ycsb_tbl<block-remote>", {"col_1"},
                  CatalogParser::getInstance(), pg{AeolusRemotePlugin::type})
            .unpack()
            .print((output_binary ? pg(bin_plugin) : pg(csv_plugin)),
                   (output_binary ? "ycsb_tbl" : "ycsb_tbl.tbl"));
    return rel.prepare();
  }

 public:
  //-- generator
  class YCSBTxnGen : public bench::BenchQueue {
    YCSBTxnGen(YCSB &ycsbBench, worker_id_t wid, partition_id_t partition_id)
        : ycsbBench(ycsbBench), wid(wid), partition_id(partition_id) {
      this->_txn_mem = ycsbBench.get_query_struct(partition_id);
    }
    ~YCSBTxnGen() override { bench::YCSB::free_query_struct(_txn_mem); }

    txn::StoredProcedure pop(worker_id_t workerId,
                             partition_id_t partitionId) override {
      ycsbBench.gen_txn(workerId, this->_txn_mem, partitionId);

      return txn::StoredProcedure(
          [this](txn::TransactionExecutor &executor, txn::Txn &txn,
                 void *params) {
            return this->ycsbBench.exec_txn(executor, txn, params);
          },
          this->_txn_mem);
    }

    void pre_run() override { ycsbBench.pre_run(wid, 0, partition_id, 0); }
    void post_run() override {}
    void dump(std::string name) override {
      if (this->wid == 0) {
        auto qry = YCSB::export_tbl(false, ycsbBench.name + name);
        LOG(INFO) << qry.execute();
      }
    }

   private:
    struct YCSB_TXN *_txn_mem;
    YCSB &ycsbBench;
    const worker_id_t wid;
    const partition_id_t partition_id;

    friend class YCSB;
  };
  //-- END generator

  friend class YCSBTxnGen;
};

}  // namespace bench

#endif /* BENCH_YCSB_HPP_ */
