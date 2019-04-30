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

#ifndef YCSB_HPP_
#define YCSB_HPP_

// extern "C" {
//#include "stdlib.h"
//}

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>
#include "benchmarks/bench.hpp"
#include "scheduler/topology.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
//#include <thread

namespace bench {

/*

        Benchmark: Yahoo! Cloud Serving Benchmark

        Description:

        Tunable Parameters:

*/

class YCSB : public Benchmark {
 private:
  const int num_fields;
  const int num_records;
  const double theta;
  const int num_iterations_per_worker;
  const int num_ops_per_txn;
  const double write_threshold;
  int num_max_workers;
  int num_active_workers;

  storage::Schema *schema;
  storage::Table *ycsb_tbl;

  std::atomic<bool> initialized;  // so that nobody calls load twice
                                  // std::atomic<uint64_t> key_gen;

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
  void load_data(int num_threads = 1) {
    std::cout << "[YCSB] Loading data.." << std::endl;
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;
    for (int i = 0; i < num_fields; i++) {
      columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
          "col_" + std::to_string(i + 1), storage::INTEGER, sizeof(uint64_t)));
    }
    ycsb_tbl = schema->create_table("ycsb_tbl", storage::COLUMN_STORE, columns,
                                    num_records);

    /* Load data into tables*/

    // multi-thread is broken
    /*uint64_t block_size = num_records / num_threads;
    std::cout << "-------------" << std::endl;
    std::cout << "------YCSB LOAD-------" << std::endl;
    std::cout << "Total Records: " << num_records << std::endl;
    std::cout << "Total Threads: " << num_threads << std::endl;
    std::cout << "Block size: " << block_size << std::endl;

    std::thread *thd_arr[num_threads];

    for (int i = 0; i < num_threads; i++) {
      uint64_t start = block_size * i;
      uint64_t end = (block_size * (i + 1)) - 1;

      uint64_t args[3];
      args[0] = start;
      args[1] = end;

      if (end > num_records) {
        end = num_records;
      }
      std::cout << "Block-" << i << " | start: " << start << ", end: " << end
                << std::endl;

      thd_arr[i] = new std::thread(&YCSB::load_data_thread, this, &args);
    }

    for (int i = 0; i < num_threads; i++) {
      thd_arr[i]->join();
      delete thd_arr[i];
    }*/

    // txn::TransactionManager *txnManager =
    //    &txn::TransactionManager::getInstance();

    for (int i = 0; i < num_records; i++) {
      std::vector<uint64_t> tmp(num_fields, i);

      struct YCSB_TXN insert_txn = gen_insert_txn(i, &tmp);
      // txnManager->executor.execute_txn(
      this->exec_txn(&insert_txn, 0, 0,
                     0);  // txn_id = 0; master= 0; delta_ver = 0;
      // txnManager->get_next_xid(0),txnManager->curr_master);
      if (i % 1000000 == 0)
        std::cout << "[YCSB] inserted records: " << i << std::endl;
    }
    std::cout << "[YCSB] inserted records: " << num_records << std::endl;
    initialized = true;
  };

  struct YCSB_TXN gen_insert_txn(uint64_t key, void *rec) {
    struct YCSB_TXN txn;

    txn.ops = new YCSB_TXN_OP[1];
    assert(txn.ops != NULL);

    txn.ops[0].op_type = txn::OPTYPE_INSERT;
    // txn.ops[0].data_table = ycsb_tbl;
    txn.ops[0].key = key;
    txn.ops[0].rec = rec;
    txn.n_ops = 1;
    return txn;
  }
  struct YCSB_TXN gen_upd_txn(uint64_t key, void *rec) {
    struct YCSB_TXN txn;

    txn.ops = new YCSB_TXN_OP[1];
    assert(txn.ops != NULL);

    txn.ops[0].op_type = txn::OPTYPE_UPDATE;
    // txn.ops[0].data_table = ycsb_tbl;
    txn.ops[0].key = key;
    txn.ops[0].rec = rec;
    txn.n_ops = 1;
    return txn;
  }

  void *get_query_struct_ptr() {
    struct YCSB_TXN *txn = new struct YCSB_TXN;
    txn->ops = new struct YCSB_TXN_OP[num_ops_per_txn];

    // struct YCSB_TXN *txn = (struct YCSB_TXN *)malloc(sizeof(struct
    // YCSB_TXN)); txn->ops = (struct YCSB_TXN_OP *)calloc(num_ops_per_txn,
    //                                         sizeof(struct YCSB_TXN_OP));
    // new YCSB_TXN_OP[num_ops_per_txn];
    return txn;
  }

  /* FIXME: Possible memory leak because we dont clear the TXN memory*/
  void *gen_txn(int wid, void *txn_ptr) {
    struct YCSB_TXN *txn = (struct YCSB_TXN *)txn_ptr;

    txn::OP_TYPE op;
    wid = wid % num_active_workers;
    // std::cout << "wid :" << wid
    //          << "| c= " << ((int)(write_threshold * (double)num_workers))
    //          << std::endl;

    if (wid >= (write_threshold * (double)num_active_workers)) {
      op = txn::OPTYPE_LOOKUP;
      // std::cout << "L ";
    } else {
      op = txn::OPTYPE_UPDATE;
      // std::cout << "U ";
    }

    // op = txn::OPTYPE_UPDATE;
    // op = txn::OPTYPE_LOOKUP;

    bool is_duplicate = false;
    for (int i = 0; i < num_ops_per_txn; i++) {
      txn->ops[i].op_type = op;
      // txn->ops[i].data_table = ycsb_tbl;
      txn->ops[i].rec = nullptr;

      do {
        // make op
        zipf_val(wid, &txn->ops[i]);
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
    return txn;
  }

  bool exec_txn(void *stmts, uint64_t xid, ushort master_ver,
                ushort delta_ver) {
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
          if (hash_ptr->write_lck.compare_exchange_strong(e_false, true)) {
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
          if (txn::CC_MV2PL::is_readable(hash_ptr->t_min, hash_ptr->t_max,
                                         xid)) {
            ycsb_tbl->touchRecordByKey(hash_ptr->VID,
                                       hash_ptr->last_master_ver);
          } else {
            void *v = ycsb_tbl->getVersions(hash_ptr->VID, delta_ver)
                          ->get_readable_ver(xid);
          }
          hash_ptr->latch.release();
          break;
        }

        case txn::OPTYPE_UPDATE: {
          global_conf::IndexVal *hash_ptr =
              (global_conf::IndexVal *)ycsb_tbl->p_index->find(op.key);

          hash_ptr->latch.acquire();
          ycsb_tbl->updateRecord(
              hash_ptr->VID, op.rec, master_ver, hash_ptr->last_master_ver,
              delta_ver, hash_ptr->t_min, hash_ptr->t_max,
              (xid >> 56) % NUM_SOCKETS);  // this is the number of sockets
          hash_ptr->t_min = xid;
          hash_ptr->last_master_ver = master_ver;
          hash_ptr->latch.release();
          break;
        }
        case txn::OPTYPE_INSERT: {
          void *hash_ptr = ycsb_tbl->insertRecord(op.rec, xid, master_ver);
          ycsb_tbl->p_index->insert(op.key, hash_ptr);
          break;
        }
        default:
          break;
      }
    }

    txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired);

    return true;
  }

  // TODO: clean-up
  ~YCSB() { std::cout << "destructor of YCSB" << std::endl; }

  // private:
  YCSB(std::string name = "YCSB", int num_fields = 2, int num_records = 1000000,
       double theta = 0.5, int num_iterations_per_worker = 1000000,
       int num_ops_per_txn = 2, double write_threshold = 0.5,
       int num_active_workers = -1, int num_max_workers = -1)
      : Benchmark(name),
        num_fields(num_fields),
        num_records(num_records),
        theta(theta),
        num_iterations_per_worker(num_iterations_per_worker),
        num_ops_per_txn(num_ops_per_txn),
        write_threshold(write_threshold),
        num_max_workers(num_max_workers),
        num_active_workers(num_active_workers) {
    initialized = false;
    // num_workers = scheduler::Topology::getInstance().get_num_worker_cores();
    if (num_max_workers == -1)
      num_max_workers = std::thread::hardware_concurrency();
    if (num_active_workers == -1)
      num_active_workers = std::thread::hardware_concurrency();
    // key_gen = 0;
    init();

    std::cout << "W_THRESH: " << write_threshold << std::endl;

    for (int i = 0; i < num_active_workers; i++) {
      std::cout << "WID: " << i;
      if (i >= (write_threshold * (double)num_active_workers)) {
        std::cout << " L ";
      } else {
        std::cout << " U ";
      }
      std::cout << std::endl;
    }

    this->schema = &storage::Schema::getInstance();
  };

  struct drand48_data *rand_buffer;
  double g_zetan;
  double g_zeta2;
  double g_eta;
  double g_alpha_half_pow;

  void init() {
    printf("Initializing zipf\n");
    rand_buffer = (struct drand48_data *)calloc(num_max_workers,
                                                sizeof(struct drand48_data));

    for (int i = 0; i < num_max_workers; i++) {
      srand48_r(i + 1, &rand_buffer[i]);
    }

    uint64_t n = num_records - 1;
    g_zetan = zeta(n, theta);
    g_zeta2 = zeta(2, theta);

    g_eta = (1 - pow(2.0 / n, 1 - theta)) / (1 - g_zeta2 / g_zetan);
    g_alpha_half_pow = 1 + pow(0.5, theta);
    printf("n = %lu\n", n);
    printf("theta = %.2f\n", theta);
  }

  inline void zipf_val(int wid, struct YCSB_TXN_OP *op) {
    uint64_t n = num_records - 1;

    // elasticity hack when we will increase num_server on runtime
    // wid = wid % num_workers;

    double alpha = 1 / (1 - theta);

    double u;
    drand48_r(&rand_buffer[wid], &u);
    double uz = u * g_zetan;

    if (uz < 1) {
      op->key = 0;
    } else if (uz < g_alpha_half_pow) {
      op->key = 1;
    } else {
      op->key = (uint64_t)(n * pow(g_eta * u - g_eta + 1, alpha));
    }

    // get the server id for the key
    int tserver = op->key % num_max_workers;
    // get the key count for the key
    uint64_t key_cnt = op->key / num_max_workers;

    uint64_t recs_per_server = num_records / num_max_workers;
    op->key = tserver * recs_per_server + key_cnt;

    assert(op->key < num_records);
  }

  inline double zeta(uint64_t n, double theta) {
    double sum = 0;
    for (uint64_t i = 1; i <= n; i++) sum += std::pow(1.0 / i, theta);
    return sum;
  }
};

}  // namespace bench

#endif /* YCSB_HPP_ */
