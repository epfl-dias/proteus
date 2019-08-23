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

#endif

#define LINEORDER_EXTRA_RESERVE 0

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
  struct ssb_query {};

  void create_tbl_part(uint64_t num_part = NUM_PARTS);
  void create_tbl_lineorder(uint64_t num_lo = INITAL_NUM_LINEORDER);

  void load_data(int num_threads = 1);  // interface

  void load_part(uint64_t num_part = NUM_PARTS);
  void load_lineorder(uint64_t num_lo = INITAL_NUM_LINEORDER);

  void *get_query_struct_ptr() { return new struct ssb_query; }  // interface

  void get_next_neworder_query(void *arg);

  bool exec_txn(void *stmts, uint64_t xid, ushort master_ver, ushort delta_ver);
  void gen_txn(int wid, void *txn_ptr);

  ~MicroSSB();
  MicroSSB(std::string name = "MicroSSB",
           std::string binary_path_root = DATA_PATH);
};

}  // namespace bench

#endif /* BENCH_MICRO_SSB_HPP_ */
