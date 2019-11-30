/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef BENCH_YCSB_HPP_
#define BENCH_YCSB_HPP_

// extern "C" {
//#include "stdlib.h"
//}

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

#include "glo.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/topology.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
//#include <thread

namespace bench {

/*
  FIXME: zipfian is broken in YCSB.
*/

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
  uint64_t recs_per_server;
  storage::Schema *schema;
  storage::Table *ycsb_tbl;

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
    short n_ops;

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

    // static std::mutex out_lk;
    // {
    //   std::unique_lock<std::mutex> lk(out_lk);
    //   std::cout << "Worker-" << wid << " : Inserted[" << partition_id << "] "
    //             << to_ins << " records (" << start << " -- " << (start +
    //             to_ins)
    //             << ")" << std::endl;
    // }

    free_query_struct_ptr(q_ptr);
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
    storage::MemoryManager::free(txn->ops);
    //,sizeof(struct YCSB_TXN_OP) * num_ops_per_txn);
    storage::MemoryManager::free(txn);  //, sizeof(struct YCSB_TXN));
  }

  void *get_query_struct_ptr(ushort pid) {
    // struct YCSB_TXN *txn = new struct YCSB_TXN;
    // txn->ops = new struct YCSB_TXN_OP[num_ops_per_txn];
    // return txn;

    struct YCSB_TXN *txn = (struct YCSB_TXN *)storage::MemoryManager::alloc(
        sizeof(struct YCSB_TXN), pid, MADV_DONTFORK);
    txn->ops = (struct YCSB_TXN_OP *)storage::MemoryManager::alloc(
        sizeof(struct YCSB_TXN_OP) * num_ops_per_txn, pid, MADV_DONTFORK);
    return txn;
  }

  void gen_txn(int wid, void *txn_ptr, ushort partition_id) {
    struct YCSB_TXN *txn = (struct YCSB_TXN *)txn_ptr;

#if YCSB_MIXED_OPS

    uint num_w_ops = write_threshold * num_ops_per_txn;

#else

    txn::OP_TYPE op;
    ushort wid_n = wid % num_active_workers;

    if (wid_n >= (write_threshold * (double)num_active_workers)) {
      op = txn::OPTYPE_LOOKUP;
    } else {
      op = txn::OPTYPE_UPDATE;
    }

#endif
    // op = txn::OPTYPE_UPDATE;
    // op = txn::OPTYPE_LOOKUP;

    bool is_duplicate = false;
    for (int i = 0; i < num_ops_per_txn; i++) {
#if YCSB_MIXED_OPS

      if (i < num_w_ops) {
        txn->ops[i].op_type = txn::OPTYPE_UPDATE;
      } else {
        txn->ops[i].op_type = txn::OPTYPE_LOOKUP;
      }

#else
      txn->ops[i].op_type = op;

#endif

      // txn->ops[i].data_table = ycsb_tbl;
      txn->ops[i].rec = nullptr;

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
    }
    txn->n_ops = num_ops_per_txn;
  }

  bool exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                ushort delta_ver, ushort partition_id) {
    struct YCSB_TXN *txn_stmts = (struct YCSB_TXN *)stmts;
    int n = txn_stmts->n_ops;

    /* Acquire locks for updates*/

    std::vector<global_conf::IndexVal *> hash_ptrs_lock_acquired;
    for (int i = 0; i < n; i++) {
      struct YCSB_TXN_OP op = txn_stmts->ops[i];

      switch (op.op_type) {
        case txn::OPTYPE_UPDATE: {
          global_conf::IndexVal *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          bool e_false = false;
          if (hash_ptr->write_lck.try_lock()) {
            hash_ptrs_lock_acquired.emplace_back(hash_ptr);
          } else {
            txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired);
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
          if (txn::CC_MV2PL::is_readable(hash_ptr->t_min, xid)) {
            ycsb_tbl->touchRecordByKey(hash_ptr->VID);
          } else {
            // in order to put an assert on valid list, the current tag resides
            // in delta-store, so this should come from delta store, but for now
            // lets assume correctness and directly access the list.
            // void *v =
            //     ycsb_tbl->getVersions(hash_ptr->VID)->get_readable_ver(xid);

            void *v = hash_ptr->delta_ver->get_readable_ver(xid);
          }
          hash_ptr->latch.release();
          break;
        }

        case txn::OPTYPE_UPDATE: {
          global_conf::IndexVal *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          hash_ptr->latch.acquire();

          ycsb_tbl->updateRecord(hash_ptr, op.rec, master_ver, delta_ver);
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

    // txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired);

    return true;
  }

  // TODO: clean-up
  ~YCSB() { std::cout << "destructor of YCSB" << std::endl; }

  // private:
  YCSB(std::string name = "YCSB", int num_fields = 2, int num_records = 1000000,
       double theta = 0.5, int num_iterations_per_worker = 1000000,
       int num_ops_per_txn = 2, double write_threshold = 0.5,
       int num_active_workers = -1, int num_max_workers = -1,
       ushort num_partitions = 1, bool layout_column_store = true)
      : Benchmark(name, num_active_workers, num_max_workers, num_partitions),
        num_fields(num_fields),
        num_records(num_records),
        theta(theta),
        // num_iterations_per_worker(num_iterations_per_worker),
        num_ops_per_txn(num_ops_per_txn),
        write_threshold(write_threshold) {
    // num_workers = scheduler::Topology::getInstance().get_num_worker_cores();
    if (num_max_workers == -1)
      num_max_workers = std::thread::hardware_concurrency();
    if (num_active_workers == -1)
      num_active_workers = std::thread::hardware_concurrency();

    assert(this->num_records % this->num_max_workers == 0 &&
           "Total number of records should be divisible by total # cores");
    this->recs_per_server = this->num_records / this->num_max_workers;
    this->schema = &storage::Schema::getInstance();
    std::cout << "workers: " << this->num_active_workers << std::endl;
    init();

    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;
    for (int i = 0; i < num_fields; i++) {
      columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
          "col_" + std::to_string(i + 1), storage::INTEGER, sizeof(uint64_t)));
    }

    ycsb_tbl = schema->create_table(
        "ycsb_tbl",
        (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
        columns, num_records);
  }

  void init() {
    printf("Initializing zipf\n");

    rand_buffer = (struct drand48_data **)calloc(num_partitions,
                                                 sizeof(struct drand48_data *));
    for (ushort i = 0; i < num_partitions; i++) {
      rand_buffer[i] = (struct drand48_data *)storage::MemoryManager::alloc(
          (num_max_workers / num_partitions) * sizeof(struct drand48_data), i,
          MADV_DONTFORK);

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
    printf("n = %lu\n", n);
    printf("theta = %.2f\n", theta);
  }

  inline void zipf_val(int wid, ushort partition_id, struct YCSB_TXN_OP *op) {
    uint64_t n = num_records - 1;

    // elasticity hack when we will increase num_server on runtime
    // wid = wid % num_workers;

    static thread_local uint64_t recs_per_server_tlocal = this->recs_per_server;
    static thread_local double g_eta_tlocal = this->g_eta;
    static thread_local double g_alpha_half_pow_tlocal = this->g_alpha_half_pow;
    static thread_local double g_zetan_tlocal = this->g_zetan;
    static thread_local double theta_tlocal = this->theta;

    double alpha = 1 / (1 - theta_tlocal);
    double u;

    drand48_r(&rand_buffer[partition_id][wid % NUM_CORE_PER_SOCKET], &u);

    double uz = u * g_zetan_tlocal;

    // std::cout << "u: " << u << std::endl;
    // std::cout << "uz: " << u << std::endl;

    if (uz < 1) {
      op->key = 0;
    } else if (uz < g_alpha_half_pow_tlocal) {
      op->key = 1;
    } else {
      op->key = (uint64_t)(n * pow(g_eta_tlocal * u - g_eta_tlocal + 1, alpha));
    }
    // std::cout << "op->key: " << op->key << std::endl;
    op->key = op->key % recs_per_server_tlocal;
    // std::cout << "op->key: " << op->key << std::endl;
    // // ---
    // uint64_t recs_per_server = num_records / num_max_workers;
    // uint tserver = op->key / recs_per_server;

    // ---
    // std::cout << "op->key: " << op->key << std::endl;

    // get the server id for the key
    // int tserver = op->key % num_max_workers;
    // std::cout << "tserver: " << tserver << std::endl;
    // get the key count for the key
    // uint64_t key_cnt = op->key / num_max_workers;
    // std::cout << "key_cnt: " << key_cnt << std::endl;
    // std::cout << "wid: " << wid << std::endl;
    // std::cout << "recs_per_server: " << recs_per_server << std::endl;
    // op->key = tserver * recs_per_server + key_cnt;
    op->key = wid * recs_per_server_tlocal + op->key;
    assert(op->key >= (wid * 1000000) && op->key < ((wid + 1) * 1000000));
    // std::cout << "op->key: " << op->key << std::endl;

    assert(op->key < num_records);
  }

  inline double zeta(uint64_t n, double theta) {
    double sum = 0;
    for (uint64_t i = 1; i <= n; i++) sum += std::pow(1.0 / i, theta);
    return sum;
  }
};

}  // namespace bench

#endif /* BENCH_YCSB_HPP_ */
