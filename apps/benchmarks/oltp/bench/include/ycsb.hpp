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
#include <oltp/common/constants.hpp>
#include <oltp/interface/bench.hpp>
#include <oltp/storage/table.hpp>
#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/topology.hpp>
#include <thread>
#include <utility>

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
 private:
  const int num_fields;
  const int num_records;
  // const int num_iterations_per_worker;
  const int num_ops_per_txn;
  const double write_threshold;

  bench_utils::ZipfianGenerator<size_t> zipf;

  // uint64_t recs_per_server;
  storage::Schema *schema;
  storage::Table *ycsb_tbl{};

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

 public:
  void pre_run(worker_id_t wid, xid_t xid, partition_id_t partition_id,
               master_version_t master_ver) override {
    assert(xid < TXN_ID_BASE);
    LOG(INFO) << "PRE_RUN_XID: " << xid;
    uint64_t to_ins = num_records / num_max_workers;
    uint64_t start = to_ins * wid;

    auto *q_ptr = (struct YCSB_TXN *)get_query_struct_ptr(partition_id);

    for (uint64_t i = start; i < (start + to_ins); i++) {
      std::vector<uint64_t> tmp(num_fields, i);
      gen_insert_txn(i, &tmp, q_ptr);
      auto pseudoTxn = txn::Txn(q_ptr, xid);
      this->exec_txn(pseudoTxn, master_ver, 0, partition_id);
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
    MemoryManager::freePinned(txn->ops);
    MemoryManager::freePinned(txn);
  }

  void *get_query_struct_ptr(partition_id_t pid) override {
    auto *txn = (struct YCSB_TXN *)MemoryManager::mallocPinnedOnNode(
        sizeof(struct YCSB_TXN), pid);
    txn->ops = (struct YCSB_TXN_OP *)MemoryManager::mallocPinnedOnNode(
        sizeof(struct YCSB_TXN_OP) * num_ops_per_txn, pid);
    return txn;
  }

  void gen_txn(worker_id_t wid, void *txn_ptr,
               partition_id_t partition_id) override {
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
    //    if (txn->n_ops == txn::OPTYPE_LOOKUP) {
    //      txn->n_ops = num_ops_per_txn;
    //    }
    //    if(op == txn::OPTYPE_LOOKUP){
    //      txn->n_ops = num_ops_per_txn;
    //    } else {
    //      txn->n_ops = num_ops_per_txn;
    //    }
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

  bool exec_txn(txn::Txn &txn, master_version_t master_ver,
                delta_id_t delta_ver, partition_id_t partition_id) override {
    return exec_txn_mv2pl(txn, master_ver, delta_ver, partition_id);
  }

  bool exec_txn_mv2pl(txn::Txn &txn, master_version_t master_ver,
                      delta_id_t delta_ver,
                      partition_id_t partition_id) override {
    // In MV2PL, txn.txnTs.txn_start_time == txn.txnTs.txn_commit_time

    auto *txn_stmts = (struct YCSB_TXN *)txn.stmts;
    int n = txn_stmts->n_ops;

    // static thread_local int n = this->num_ops_per_txn;
    static thread_local ushort num_col_upd = this->num_of_col_upd_per_op;
    static thread_local ushort num_col_read = this->num_of_col_read_per_op;
    static thread_local std::vector<column_id_t> col_idx_read_local(
        col_idx_read);
    static thread_local std::vector<column_id_t> col_idx_update_local(
        col_idx_upd);
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
          ycsb_tbl->getIndexedRecord(txn.txnTs, *hash_ptr, read_loc.data(),
                                     col_idx_read_local.data(), num_col_read);
          hash_ptr->latch.release();
          break;
        }

        case txn::OPTYPE_UPDATE: {
          auto *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          hash_ptr->latch.acquire();
          ycsb_tbl->updateRecord(txn.txnTs.txn_start_time, hash_ptr, op.rec,
                                 delta_ver, col_idx_update_local.data(),
                                 num_col_upd, master_ver);
          hash_ptr->latch.release();
          hash_ptr->write_lck.unlock();
          break;
        }
        case txn::OPTYPE_INSERT: {
          void *hash_ptr = ycsb_tbl->insertRecord(
              op.rec, txn.txnTs.txn_start_time, partition_id, master_ver);
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

  bool exec_txn_mvocc(txn::Txn &txn, master_version_t master_ver,
                      delta_id_t delta_ver,
                      partition_id_t partition_id) override {
    auto *txn_stmts = (struct YCSB_TXN *)txn.stmts;
    int n = txn_stmts->n_ops;

    // static thread_local int n = this->num_ops_per_txn;
    static thread_local ushort num_col_upd = this->num_of_col_upd_per_op;
    static thread_local ushort num_col_read = this->num_of_col_read_per_op;
    static thread_local std::vector<column_id_t> col_idx_read_local(
        col_idx_read);
    static thread_local std::vector<column_id_t> col_idx_update_local(
        col_idx_upd);
    static thread_local std::vector<uint64_t> read_loc(num_fields, 0);

    bool abort = false;
    bool read_only = true;
    int i = 0;
    assert(txn.txnTs.txn_start_time < txn.txnTs.txn_id);

    // xid = transaction_id
    // startTime = higherTs
    // commitTime = commitTs (lowerTs

    static thread_local std::vector<global_conf::IndexVal *> indexSearches(
        n, nullptr);

    for (i = 0; i < n && !abort; i++) {
      struct YCSB_TXN_OP op = txn_stmts->ops[i];

      switch (op.op_type) {
        case txn::OPTYPE_UPDATE: {
          read_only = false;
          indexSearches[i] =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          // check if somebody hasnt updated it yet.
          auto old_val = indexSearches[i]->t_min;
          if (old_val < txn.txnTs.txn_start_time) {
            // if(hash_ptr->write_lck.try_lock()){
            // nobody has overwrote this one yet.

            // CORRECT 1
            //            hash_ptr->latch.acquire();
            //            ycsb_tbl->updateRecord(txn.txnTs.txn_id, hash_ptr,
            //            op.rec,
            //                                   delta_ver,
            //                                   col_idx_update_local.data(),
            //                                   num_col_upd, master_ver);
            //            hash_ptr->t_min = txn.txnTs.txn_id;
            //            hash_ptr->latch.release();
            //            assert(hash_ptr->t_min == txn.txnTs.txn_id && "I
            //            update so it should be!!");

            // CORRECT 2
            ycsb_tbl->createVersion(txn.txnTs.txn_id, indexSearches[i],
                                    delta_ver, col_idx_update_local.data(),
                                    num_col_upd);
            // hash_ptr->t_min = txn.txnTs.txn_id;
            // auto old_val = hash_ptr->t_min;
            if (__sync_bool_compare_and_swap(&(indexSearches[i]->t_min),
                                             old_val, txn.txnTs.txn_id)) {
              ycsb_tbl->updateRecordWithoutVersion(
                  txn.txnTs.txn_id, indexSearches[i], op.rec, delta_ver,
                  col_idx_update_local.data(), num_col_upd, master_ver);
              // std::atomic_thread_fence(std::memory_order_seq_cst);
            } else {
              abort = true;
            }

            // first create version
            // then update Ts
            // then update Data

            // in this case, even someone read new Ts, goes to delta, they will
            // find something.

            // create-version
            /*ycsb_tbl->createVersion(txn.txnTs.txn_start_time,
                                    hash_ptr,
                                       delta_ver,
                                    col_idx_update_local.data(),
                                    num_col_upd);*/

            // updateTs
            // this should be compare-and-swap

            // what if someone created version but failed on C&S Ts?
            // for now lets latch it to prevent false access?
            // because the delta-list is not concurrent protected.
            // otherwise we will need a latch in writing version anyway/

            // updateData
            /*ycsb_tbl->updateRecordWithoutVersion(txn.txnTs.txn_id,
                                                    hash_ptr, op.rec,
                                                    delta_ver,
                                                    col_idx_update_local.data(),
                                                    num_col_upd,
                                                    master_ver);*/

          } else {
            // LOG(INFO) << "Abort Txn: " << txn.txnTs.txn_id << " | record: "
            // << op.key << "| i: " << i;
            abort = true;
          }

          break;
        }
        case txn::OPTYPE_LOOKUP: {
          indexSearches[i] =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          // following only compares with xid. if myself has written this record
          // then it will return wrong result.
          // hash_ptr->latch.acquire();
          auto tmin = indexSearches[i]->t_min;
          ycsb_tbl->getIndexedRecord(txn.txnTs, *indexSearches[i],
                                     read_loc.data(), col_idx_read_local.data(),
                                     num_col_read);
          // hash_ptr->latch.release();

          // compiler fence for re-order.
          //          if (__unlikely(tmin != indexSearches[i]->t_min)) {
          //            // failed, fall back to latch
          //            //indexSearches[i]->latch.acquire();
          ////            ycsb_tbl->getIndexedRecord(txn.txnTs,
          ///*indexSearches[i], read_loc.data(), / col_idx_read_local.data(),
          /// num_col_read);
          //            //indexSearches[i]->latch.release();
          //            abort = true;
          //          }

          // read TS, try reading data, and re-verify Ts, to verify if read was
          // successful? try optimistic, and if read fails, either acquire
          // latch, or abort (read-write conflict)

          // do we need the following check?
          //          if(hash_ptr->t_min < xid.txn_id || hash_ptr->t_min==
          //          xid.txn_start_time){
          //            // nobody has overwrote this one yet.
          //            // can read the top version.
          //          }

          break;
        }

        case txn::OPTYPE_INSERT: {
          read_only = false;
          void *hash_ptr = ycsb_tbl->insertRecord(
              op.rec, txn.txnTs.txn_start_time, partition_id, master_ver);
          ycsb_tbl->p_index->insert(op.key, hash_ptr);
          indexSearches[i] =
              static_cast<txn::CC_MV2PL::PRIMARY_INDEX_VAL *>(hash_ptr);
          break;
        }
        default: {
          throw std::runtime_error("YCSB: unknown op");
        }
      }
    }

    if (__unlikely(!read_only)) {
      if (__unlikely(abort)) {
        // aborted on op i, so start rollback from i-1

        for (i = i - 2; i >= 0; i--) {
          struct YCSB_TXN_OP *op = &(txn_stmts->ops[i]);

          // instead of finding again, we can also save the pointers.
          switch (op->op_type) {
            case txn::OPTYPE_UPDATE: {
              // LOG(INFO) << "Rollback Txn: " << txn.txnTs.txn_id << " |
              // record: " << op.key << "| i: " << i;
              if (__unlikely(indexSearches[i] == nullptr)) {
                indexSearches[i] =
                    (global_conf::IndexVal *)ycsb_tbl->p_index->find(op->key);
              }

              //              if(hash_ptr->t_min != txn.txnTs.txn_id){
              //                LOG(INFO) << "Mismatch: " << op.key << " :: " <<
              //                hash_ptr->t_min << " | " << txn.txnTs.txn_id;
              //              }

              // assert(indexSearches[i]->t_min == txn.txnTs.txn_id);
              // hash_ptr->latch.acquire();

              ycsb_tbl->updateRollback(txn.txnTs, indexSearches[i],
                                       col_idx_update_local.data(),
                                       num_col_upd);
              // hash_ptr->latch.release();

              // assert(indexSearches[i]->t_min < txn.txnTs.txn_start_time);

              break;
            }
            case txn::OPTYPE_INSERT: {
              LOG(INFO) << "Rollback insert txn, what to do?";
              break;
            }
            case txn::OPTYPE_LOOKUP:
            default:
              break;
          }
        }

      } else {
        auto commitTs = txn.getCommitTs();

        // validate
        // write-write conflicts are detected earlier
        // read-write conflict detection:
        //    modified object
        //    deleted object
        //    created object
        //    created-and-deleted object

        // go into commit procedure

        for (i = 0; i < n; i++) {
          struct YCSB_TXN_OP *op = &(txn_stmts->ops[i]);

          // instead of finding again, we can also save the pointers.

          //          auto *hash_ptr =
          //              (global_conf::IndexVal
          //              *)ycsb_tbl->p_index->find(op->key);

          switch (op->op_type) {
            // update the txnTs of inserted records.
            // update the txnTs of updated records.
            // also update the TS in delta store (undo-buffer)?
            // will be needed when an record is update twice in same Txn.
            case txn::OPTYPE_INSERT:
            case txn::OPTYPE_UPDATE: {
              if (__unlikely(indexSearches[i] == nullptr)) {
                indexSearches[i] =
                    (global_conf::IndexVal *)ycsb_tbl->p_index->find(op->key);
              }
              // hash_ptr->latch.acquire();
              indexSearches[i]->t_min = commitTs;
              // std::atomic_thread_fence(std::memory_order_seq_cst);
              // hash_ptr->latch.release();
              break;
            }
            case txn::OPTYPE_LOOKUP:
            default:
              break;
          }
        }
      }
    }

    return !abort;
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
    LOG(INFO) << "workers: " << (uint)(this->num_active_workers);
    LOG(INFO) << "Max-Workers: " << (uint)(this->num_max_workers);
    init();

    storage::TableDef columns;
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
