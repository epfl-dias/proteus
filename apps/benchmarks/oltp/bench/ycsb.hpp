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

#include "common/common.hpp"
#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "topology/topology.hpp"

namespace bench {

#define THREAD_LOCAL 0
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
  const double theta;
  // const int num_iterations_per_worker;
  const int num_ops_per_txn;
  const double write_threshold;

  // uint64_t recs_per_server;
  storage::Schema *schema;
  storage::Table *ycsb_tbl;

  const int num_of_col_upd_per_op;
  std::vector<ushort> col_idx;

  // zipf stuff
  struct drand48_data **rand_buffer;
  double g_zetan;
  double g_zeta2;
  double g_eta;
  double g_alpha_half_pow;

  struct YCSB_TXN_OP {
    uint64_t key;
    txn::OP_TYPE op_type;
    void *rec;
  };  // __attribute__((aligned(64)));

  struct YCSB_TXN {
    struct YCSB_TXN_OP *ops;
    uint n_ops;

    ~YCSB_TXN() { delete ops; }
  };  // __attribute__((aligned(64)));

 public:
  void pre_run(int wid, uint64_t xid, ushort partition_id, ushort master_ver) {
    uint64_t to_ins = num_records / num_max_workers;
    uint64_t start = to_ins * wid;

    struct YCSB_TXN *q_ptr =
        (struct YCSB_TXN *)get_query_struct_ptr(partition_id);

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

  void load_data(int num_threads = 1) { assert(false && "Not implemented"); }

  void gen_insert_txn(uint64_t key, void *rec, struct YCSB_TXN *q_ptr) {
    q_ptr->ops[0].op_type = txn::OPTYPE_INSERT;
    // txn.ops[0].data_table = ycsb_tbl;
    q_ptr->ops[0].key = key;
    q_ptr->ops[0].rec = rec;
    q_ptr->n_ops = 1;
  }
  void gen_upd_txn(uint64_t key, void *rec, struct YCSB_TXN *q_ptr) {
    q_ptr->ops[0].op_type = txn::OPTYPE_UPDATE;
    // txn.ops[0].data_table = ycsb_tbl;
    q_ptr->ops[0].key = key;
    q_ptr->ops[0].rec = rec;
    q_ptr->n_ops = 1;
  }

  void free_query_struct_ptr(void *ptr) {
    struct YCSB_TXN *txn = (struct YCSB_TXN *)ptr;
    storage::memory::MemoryManager::free(txn->ops);
    storage::memory::MemoryManager::free(txn);
  }

  void *get_query_struct_ptr(ushort pid) {
    auto *txn = (struct YCSB_TXN *)storage::memory::MemoryManager::alloc(
        sizeof(struct YCSB_TXN), pid, MADV_DONTFORK);
    txn->ops = (struct YCSB_TXN_OP *)storage::memory::MemoryManager::alloc(
        sizeof(struct YCSB_TXN_OP) * num_ops_per_txn, pid, MADV_DONTFORK);
    return txn;
  }

  void gen_txn(int wid, void *txn_ptr, ushort partition_id) {
    struct YCSB_TXN *txn = (struct YCSB_TXN *)txn_ptr;

    static thread_local auto recs_per_thread =
        this->num_records / this->num_active_workers;
    static thread_local uint64_t rec_key_iter = 0;
#if YCSB_MIXED_OPS

    static thread_local uint num_w_ops = write_threshold * num_ops_per_txn;
    bool is_duplicate = false;
#else

    txn::OP_TYPE op;
    ushort wid_n = wid % num_active_workers;

    if (wid_n >= (write_threshold * (double)num_active_workers)) {
      op = txn::OPTYPE_LOOKUP;
    } else {
      op = txn::OPTYPE_UPDATE;
    }

#endif
    txn->n_ops = num_ops_per_txn;
    for (int i = 0; i < num_ops_per_txn; i++) {
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
        zipf_val(wid, partition_id, &txn->ops[i]);
        is_duplicate = false;

        for (int j = 0; j < i; j++) {
          if (txn->ops[i].key == txn->ops[j].key) {
            is_duplicate = true;
            break;
          }
        }
      } while (is_duplicate == true);
#endif
    }
  }

  bool exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                ushort delta_ver, ushort partition_id) {
    struct YCSB_TXN *txn_stmts = (struct YCSB_TXN *)stmts;
    int n = txn_stmts->n_ops;

    // static thread_local int n = this->num_ops_per_txn;
    // TODO: which cols to update?
    static thread_local int num_col_upd = this->num_of_col_upd_per_op;
    static thread_local std::vector<ushort> col_idx_local(col_idx);
    static thread_local std::vector<uint64_t> read_loc(num_fields, 0);

    static thread_local std::vector<global_conf::IndexVal *>
        hash_ptrs_lock_acquired(this->num_ops_per_txn, nullptr);
    uint num_locks = 0;

    /* Acquire locks for updates*/
    for (int i = 0; i < n; i++) {
      struct YCSB_TXN_OP op = txn_stmts->ops[i];

      switch (op.op_type) {
        case txn::OPTYPE_UPDATE: {
          global_conf::IndexVal *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          bool e_false = false;
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
          global_conf::IndexVal *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          hash_ptr->latch.acquire();

          ycsb_tbl->getRecordByKey(hash_ptr, xid, delta_ver, nullptr, 0,
                                   read_loc.data());
          hash_ptr->latch.release();
          break;
        }

        case txn::OPTYPE_UPDATE: {
          global_conf::IndexVal *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          hash_ptr->latch.acquire();
          ycsb_tbl->updateRecord(hash_ptr, op.rec, master_ver, delta_ver,
                                 col_idx_local.data(), num_col_upd);

          hash_ptr->t_min = xid;
          hash_ptr->write_lck.unlock();
          hash_ptr->latch.release();
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

  // TODO: clean-up
  ~YCSB() { std::cout << "destructor of YCSB" << std::endl; }

  // private:
  YCSB(std::string name = "YCSB", int num_fields = 2, int num_records = 1000000,
       double theta = 0.5, int num_iterations_per_worker = 1000000,
       int num_ops_per_txn = 2, double write_threshold = 0.5,
       int num_active_workers = -1, int num_max_workers = -1,
       ushort num_partitions = 1, bool layout_column_store = true,
       int num_of_col_upd = 1)
      : Benchmark(name, num_active_workers, num_max_workers, num_partitions),
        num_fields(num_fields),
        num_records(num_records),
        theta(theta),
        // num_iterations_per_worker(num_iterations_per_worker),
        num_ops_per_txn(num_ops_per_txn),
        write_threshold(write_threshold),
        num_of_col_upd_per_op(num_of_col_upd) {
    if (num_max_workers == -1)
      num_max_workers = topology::getInstance().getCoreCount();
    if (num_active_workers == -1)
      num_active_workers = topology::getInstance().getCoreCount();

    assert(num_of_col_upd_per_op <= num_fields);
    assert(this->num_records % this->num_max_workers == 0 &&
           "Total number of records should be divisible by total # cores");

    for (ushort t = 0; t < num_of_col_upd_per_op; t++) {
      col_idx.emplace_back(t);
    }
    assert(col_idx.size() == num_of_col_upd_per_op);

    // this->recs_per_server = this->num_records / this->num_max_workers;
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

  void deinit() {
    for (ushort i = 0; i < num_partitions; i++) {
      storage::memory::MemoryManager::free(rand_buffer[i]);
    }

    free(rand_buffer);
  }

  void init() {
    printf("Initializing zipf\n");

    rand_buffer = (struct drand48_data **)calloc(num_partitions,
                                                 sizeof(struct drand48_data *));
    for (ushort i = 0; i < num_partitions; i++) {
      rand_buffer[i] =
          (struct drand48_data *)storage::memory::MemoryManager::alloc(
              (num_max_workers / num_partitions) * sizeof(struct drand48_data),
              i, MADV_DONTFORK);

      // rand_buffer[i] = (struct drand48_data *)calloc(
      //   num_max_workers / num_partitions, sizeof(struct drand48_data));
    }

    int c = 0;
    for (ushort i = 0; i < num_partitions; i++) {
      for (int j = 0; j < (num_max_workers / num_partitions); j++) {
        srand48_r(c++, &rand_buffer[i][j]);
      }
    }

    uint64_t n = num_records - 1;
    g_zetan = zeta(n, theta);
    g_zeta2 = zeta(2, theta);

    g_eta = (1 - pow(2.0 / n, 1 - theta)) / (1 - g_zeta2 / g_zetan);
    g_alpha_half_pow = 1 + pow(0.5, theta);
    printf("n = %lu\n", n + 1);
    printf("theta = %.2f\n", theta);
  }

  inline void zipf_val(int wid, ushort partition_id, struct YCSB_TXN_OP *op) {
    uint64_t n = num_records - 1;

    // elasticity hack when we will increase num_server on runtime
    // wid = wid % num_workers;

    // copying class variables as thread_local variables in order to remove
    // false sharing of cachelines
    // static thread_local uint64_t recs_per_server_tlocal =
    // this->recs_per_server;
    static thread_local auto g_eta_tlocal = this->g_eta;
    static thread_local auto g_alpha_half_pow_tlocal = this->g_alpha_half_pow;
    static thread_local auto g_zetan_tlocal = this->g_zetan;
    static thread_local auto theta_tlocal = this->theta;
    static thread_local auto max_worker_per_partition =
        this->num_max_workers / this->num_partitions;

    static thread_local auto num_workers = this->num_active_workers;
    static thread_local auto g_nrecs = this->num_records;

    static thread_local double alpha = 1 / (1 - theta_tlocal);
    double u;

    drand48_r(&rand_buffer[partition_id][wid % max_worker_per_partition], &u);

    double uz = u * g_zetan_tlocal;

    if (uz < 1) {
      op->key = 0;
    } else if (uz < g_alpha_half_pow_tlocal) {
      op->key = 1;
    } else {
      op->key = (uint64_t)(n * pow(g_eta_tlocal * u - g_eta_tlocal + 1, alpha));
    }

    // Aunn's comment: i still dont understand what is the need for following
    // calculations.

    // Trireme
    //------------
    // get the server id for the key
    uint tserver = op->key % num_workers;
    // get the key count for the key
    uint64_t key_cnt = op->key / num_workers;

    uint64_t recs_per_server = g_nrecs / num_workers;
    op->key = tserver * recs_per_server + key_cnt;

    assert(op->key < g_nrecs);

    // End trireme
  }

  inline double zeta(uint64_t n, double theta_z) {
    double sum = 0;
    for (uint64_t i = 1; i <= n; i++) sum += std::pow(1.0 / i, theta_z);
    return sum;
  }
};

}  // namespace bench

#endif /* BENCH_YCSB_HPP_ */
