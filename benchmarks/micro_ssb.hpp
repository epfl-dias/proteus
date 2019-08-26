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

#ifndef BENCH_MICRO_SSB_HPP_
#define BENCH_MICRO_SSB_HPP_

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "indexes/hash_index.hpp"
#include "interfaces/bench.hpp"
#include "scheduler/topology.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
//#include <thread

#define SF 100

#if SF == 100

#define DATA_PATH "/scratch/data/ssbm100/"
#define NUM_PARTS 1400000
#define INITAL_NUM_LINEORDER 600038145
#define NUM_CUSTOMERS 89031
#define NUM_SUPPLIERS 1806

#endif

#define LINEORDER_EXTRA_RESERVE 600000000

// Query semantics

#define MICRO_SSB_MAX_PART_PER_ORDER 10

namespace bench {

class MicroSSB : public Benchmark {
 private:
  storage::Schema *schema;

  // Tables
  storage::Table *table_lineorder;
  // storage::Table *table_customer;
  // storage::Table *table_supplier;
  storage::Table *table_part;
  // storage::Table *table_date;

  std::string data_path;
  unsigned int seed;

 public:
  struct lineorder {
    uint32_t lo_orderkey;
    uint32_t lo_linenumber;
    uint32_t lo_custkey;
    uint32_t lo_partkey;
    uint32_t lo_suppkey;
    uint32_t lo_orderdate;
    // char lo_orderpriority[12];
    // char lo_shippriority[12];
    uint32_t lo_quantity;
    uint32_t lo_extendedprice;
    uint32_t lo_ordtotalprice;
    uint32_t lo_discount;
    uint32_t lo_revenue;
    uint32_t lo_supplycost;
    uint32_t lo_tax;
    uint32_t lo_commitdate;
    // char lo_shipmode[12];
  };

  struct part {
    uint32_t p_partkey;
    uint32_t p_stocklevel;  // added for upds.
    // char p_name[12];
    // char p_mfgr[12];
    // char p_category[12];
    // char p_brand1[12];
    uint32_t p_brand;
    // char p_color[12];
    // char p_type[12];
    uint32_t p_size;
    // char p_container[12];
  };

  // struct customer {
  //   uint32_t c_custkey;
  //   char c_name[12];
  //   char c_address[12];
  //   char c_city[12];
  //   char c_nation[12];
  //   char c_region[12];
  //   char c_phone[12];
  //   char c_mktsegment[12];
  // };

  // struct supplier {
  //   uint32_t s_suppkey;
  //   char s_name[12];
  //   char s_address[12];
  //   char s_city[12];
  //   char s_nation[12];
  //   char s_region[12];
  //   char s_phone[12];
  // };

  // struct date {};

  // neworder ssb query
  struct ssb_query_part {
    uint32_t partkey;
    uint32_t quantity;
    void *idx_ptr;
  };
  struct ssb_query {
    uint32_t orderkey;
    uint32_t custkey;
    uint32_t suppkey;
    uint32_t ol_cnt;
    struct ssb_query_part parts[MICRO_SSB_MAX_PART_PER_ORDER];

    // uint32_t orderpriority;
    // uint32_t shippriority;
  };

  struct drand48_data *rand_buffer;
  double g_zetan;
  double g_zeta2;
  double g_eta;
  double g_alpha_half_pow;
  double theta;
  int num_max_workers;
  uint num_records;

  void create_tbl_part(uint64_t num_part = NUM_PARTS);
  void create_tbl_lineorder(uint64_t num_lo = INITAL_NUM_LINEORDER);

  void load_data(int num_threads = 1);  // interface

  void load_part(uint64_t num_part = NUM_PARTS);
  void load_lineorder(uint64_t num_lo = INITAL_NUM_LINEORDER);
  void print_query(struct ssb_query *q);

  void *get_query_struct_ptr() { return new struct ssb_query; }  // interface

  bool exec_txn(void *stmts, uint64_t xid, ushort master_ver, ushort delta_ver);
  void gen_txn(int wid, void *txn_ptr);

  ~MicroSSB();
  MicroSSB(std::string name = "MicroSSB",
           std::string binary_path_root = DATA_PATH);

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

  inline void zipf_val(int wid, uint32_t &key) {
    uint64_t n = num_records - 1;

    // elasticity hack when we will increase num_server on runtime
    // wid = wid % num_workers;

    double alpha = 1 / (1 - theta);

    double u;
    drand48_r(&rand_buffer[wid], &u);
    double uz = u * g_zetan;

    if (uz < 1) {
      key = 0;
    } else if (uz < g_alpha_half_pow) {
      key = 1;
    } else {
      key = (uint64_t)(n * pow(g_eta * u - g_eta + 1, alpha));
    }

    // get the server id for the key
    int tserver = key % num_max_workers;
    // get the key count for the key
    uint64_t key_cnt = key / num_max_workers;

    uint64_t recs_per_server = num_records / num_max_workers;
    key = tserver * recs_per_server + key_cnt;

    assert(key < num_records);
  }

  inline double zeta(uint64_t n, double theta) {
    double sum = 0;
    for (uint64_t i = 1; i <= n; i++) sum += std::pow(1.0 / i, theta);
    return sum;
  }
};

}  // namespace bench

#endif /* BENCH_MICRO_SSB_HPP_ */
