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
#include <thread>
#include <utility>

#include "common/common.hpp"
#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "topology/topology.hpp"
#include "zipf.hpp"

namespace bench {

#define THREAD_LOCAL false
#define PARTITION_LOCAL false
#define YCSB_MIXED_OPS 0

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
 private:
  const int num_fields;
  const int num_records;
  // const int num_iterations_per_worker;
  const int num_ops_per_txn;
  const double write_threshold;

  bench_utils::ZipfianGenerator<size_t> zipf;

  // uint64_t recs_per_server;
  storage::Schema *schema;
  storage::Table *ycsb_tbl;

  const uint num_of_col_upd_per_op;
  const uint num_of_col_read_per_op;
  const uint num_col_read_offset_per_op;

  std::vector<ushort> col_idx_upd;
  std::vector<ushort> col_idx_read;

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

 public:
  void pre_run(int wid, uint64_t xid, ushort partition_id,
               ushort master_ver) override {
    uint64_t to_ins = num_records / num_max_workers;
    uint64_t start = to_ins * wid;

    auto *q_ptr = (struct YCSB_TXN *)get_query_struct_ptr(partition_id);

    for (uint64_t i = start; i < (start + to_ins); i++) {
      std::vector<uint64_t> tmp(num_fields, i);
      gen_insert_txn(i, &tmp, q_ptr);
      this->exec_txn(q_ptr, xid, master_ver, 0, partition_id);
    }
    free_query_struct_ptr(q_ptr);

    //    LOG(INFO) << "################";
    //    LOG(INFO)<< "\n" << *(ycsb_tbl->p_index);
    //    LOG(INFO) << "################";
  }

  void load_data(int num_threads) override {
    assert(false && "Not implemented");
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

  void free_query_struct_ptr(void *ptr) override {
    auto *txn = (struct YCSB_TXN *)ptr;
    storage::memory::MemoryManager::free(txn->ops);
    storage::memory::MemoryManager::free(txn);
  }

  void *get_query_struct_ptr(ushort pid) override {
    auto *txn = (struct YCSB_TXN *)storage::memory::MemoryManager::alloc(
        sizeof(struct YCSB_TXN), pid, MADV_DONTFORK);
    txn->ops = (struct YCSB_TXN_OP *)storage::memory::MemoryManager::alloc(
        sizeof(struct YCSB_TXN_OP) * num_ops_per_txn, pid, MADV_DONTFORK);
    return txn;
  }

  void gen_txn(int wid, void *txn_ptr, ushort partition_id) override {
    auto *txn = (struct YCSB_TXN *)txn_ptr;
    // txn->read_only = false;

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
    if (txn->n_ops == txn::OPTYPE_LOOKUP) {
      txn->n_ops = num_ops_per_txn * 5;
    }
    //    if(op == txn::OPTYPE_LOOKUP){
    //      txn->n_ops = num_ops_per_txn;
    //    } else {
    //      txn->n_ops = num_ops_per_txn;
    //    }
    for (int i = 0; i < txn->n_ops; i++) {
      // txn->ops[i].data_table = ycsb_tbl;
      txn->ops[i].rec = nullptr;

#if YCSB_MIXED_OPS
      if (i < num_w_ops) {
        txn->ops[i].op_type = txn::OPTYPE_UPDATE;
      } else {
        txn->ops[i].op_type = txn::OPTYPE_LOOKUP;
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

  bool exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                ushort delta_ver, ushort partition_id) override {
    auto *txn_stmts = (struct YCSB_TXN *)stmts;
    int n = txn_stmts->n_ops;

    // static thread_local int n = this->num_ops_per_txn;
    static thread_local ushort num_col_upd = this->num_of_col_upd_per_op;
    static thread_local ushort num_col_read = this->num_of_col_read_per_op;
    static thread_local std::vector<ushort> col_idx_read_local(col_idx_read);
    static thread_local std::vector<ushort> col_idx_update_local(col_idx_upd);
    static thread_local std::vector<uint64_t> read_loc(num_fields, 0);

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

    // perform lookups/ updates / inserts
    for (int i = 0; i < n; i++) {
      struct YCSB_TXN_OP op = txn_stmts->ops[i];
      switch (op.op_type) {
        case txn::OPTYPE_LOOKUP: {
          auto *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          hash_ptr->latch.acquire();

          ycsb_tbl->getRecordByKey(hash_ptr, xid, delta_ver,
                                   col_idx_read_local.data(), num_col_read,
                                   read_loc.data());
          hash_ptr->latch.release();
          break;
        }

        case txn::OPTYPE_UPDATE: {
          auto *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          hash_ptr->latch.acquire();
          ycsb_tbl->updateRecord(xid, hash_ptr, op.rec, master_ver, delta_ver,
                                 col_idx_update_local.data(), num_col_upd);
          hash_ptr->latch.release();
          hash_ptr->write_lck.unlock();
          break;
        }
        case txn::OPTYPE_INSERT: {
          void *hash_ptr =
              ycsb_tbl->insertRecord(op.rec, xid, partition_id, master_ver);
          ycsb_tbl->p_index->insert(op.key, hash_ptr);
          break;
        }
        default:
          break;
      }
    }
    // txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired.data(), num_locks);

    return true;
  }

  void deinit() override { zipf.~ZipfianGenerator(); }

  ~YCSB() override = default;

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
      num_active_workers = topology::getInstance().getCoreCount();

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
      LOG(INFO) << "AA: " << t;
      col_idx_read.emplace_back(t);
    }

    this->schema = &storage::Schema::getInstance();
    LOG(INFO) << "workers: " << this->num_active_workers;
    LOG(INFO) << "Max-Workers: " << this->num_max_workers;
    init();

    storage::ColumnDef columns;
    for (int i = 0; i < num_fields; i++) {
      columns.emplace_back("col_" + std::to_string(i + 1), storage::INTEGER,
                           sizeof(uint64_t));
    }
    ycsb_tbl = schema->create_table(
        "ycsb_tbl",
        (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
        columns, num_records);
  }
};

}  // namespace bench

#endif /* BENCH_YCSB_HPP_ */
