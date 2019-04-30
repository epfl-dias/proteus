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

#ifndef TPCC_HPP_
#define TPCC_HPP_

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <thread>
#include "benchmarks/bench.hpp"
#include "benchmarks/bench_utils.hpp"
#include "indexes/hash_index.hpp"
#include "scheduler/topology.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
//#include <thread

#define FIRST_NAME_MIN_LEN 8
#define FIRST_NAME_LEN 16
#define LAST_NAME_LEN 16
#define TPCC_MAX_ITEMS 100000
#define TPCC_NDIST_PER_WH 10
#define TPCC_NCUST_PER_DIST 3000
#define TPCC_MAX_OL_PER_ORDER 15

#define MAKE_DIST_KEY(w, d) (w * TPCC_NDIST_PER_WH + d)
#define MAKE_CUST_KEY(w, d, c) (MAKE_DIST_KEY(w, d) * TPCC_NCUST_PER_DIST + c)
#define MAKE_OL_KEY(w, d, o, ol) \
  (MAKE_CUST_KEY(w, d, o) * TPCC_MAX_OL_PER_ORDER + ol)
//#define MAKE_STOCK_KEY(w,s) (w * TPCC_MAX_ITEMS + s)

namespace bench {

/*
        Benchmark: TPC-C
*/

class TPCC : public Benchmark {
 private:
  storage::Schema *schema;
  storage::Table *table_warehouse;
  storage::Table *table_district;
  storage::Table *table_customer;
  storage::Table *table_history;
  storage::Table *table_new_order;
  storage::Table *table_order;
  storage::Table *table_order_line;
  storage::Table *table_item;
  storage::Table *table_stock;

  int num_warehouse;
  unsigned int seed;

 public:
  struct tpcc_stock {
    uint32_t s_i_id;
    ushort s_w_id;
    short s_quantity;
    char s_dist[TPCC_NDIST_PER_WH][24];
    ushort s_ytd;
    ushort s_order_cnt;
    ushort s_remote_cnt;
    char s_data[51];
  };

  struct tpcc_item {
    uint32_t i_id;
    uint32_t i_im_id;
    char i_name[25];
    float i_price;
    char i_data[51];
  };

  struct tpcc_warehouse {
    short w_id;
    char w_name[11];
    char w_street[2][21];
    char w_city[21];
    char w_state[2];
    char w_zip[9];
    float w_tax;
    float w_ytd;
  };

  struct tpcc_district {
    ushort d_id;
    ushort d_w_id;
    char d_name[11];
    char d_street[2][21];
    char d_city[21];
    char d_state[2];
    char d_zip[9];
    float d_tax;
    float d_ytd;
    uint64_t d_next_o_id;
  };
  struct tpcc_history {
    uint32_t h_c_id;
    ushort h_c_d_id;
    ushort h_c_w_id;
    ushort h_d_id;
    ushort h_w_id;
    uint32_t h_date;
    float h_amount;
    char h_data[25];
  };
  struct tpcc_customer {
    uint32_t c_id;
    ushort c_d_id;
    ushort c_w_id;
    char c_first[FIRST_NAME_LEN + 1];
    char c_middle[2];
    char c_last[LAST_NAME_LEN + 1];
    char c_street[2][21];
    char c_city[21];
    char c_state[2];
    char c_zip[9];
    char c_phone[16];
    uint32_t c_since;
    char c_credit[2];
    float c_credit_lim;
    float c_discount;
    float c_balance;
    float c_ytd_payment;
    ushort c_payment_cnt;
    ushort c_delivery_cnt;
    char c_data[501];
  };

  struct tpcc_order {
    uint64_t o_id;
    ushort o_d_id;
    ushort o_w_id;
    uint32_t o_c_id;
    uint32_t o_entry_d;
    short o_carrier_id;
    ushort o_ol_cnt;
    ushort o_all_local;
  };

  struct tpcc_order_line {
    uint64_t ol_o_id;
    ushort ol_d_id;
    ushort ol_w_id;
    ushort ol_number;
    ushort ol_i_id;
    ushort ol_supply_w_id;
    uint32_t ol_delivery_d;
    ushort ol_quantity;
    float ol_amount;
    char ol_dist_info[24];
  };
  struct tpcc_new_order {
    uint64_t no_o_id;
    ushort no_d_id;
    ushort no_w_id;
  };

  struct secondary_record {
    int sr_idx;
    int sr_nids;
    uint32_t *sr_rids;
#define NDEFAULT_RIDS 16
  };

  // fucking shortcut
  indexes::HashIndex<uint64_t, struct secondary_record> *cust_sec_index;

  void create_tbl_warehouse(uint64_t num_warehouses);
  void create_tbl_district(uint64_t num_districts);
  void create_tbl_customer(uint64_t num_cust);
  void create_tbl_history(uint64_t num_history);
  void create_tbl_new_order(uint64_t num_new_order);
  void create_tbl_order(uint64_t num_order);
  void create_tbl_order_line(uint64_t num_order_line);
  void create_tbl_item(uint64_t num_item);
  void create_tbl_stock(uint64_t num_stock);

  void load_data(int num_threads = 1);
  void load_stock(int w_id);
  void load_item(int w_id);
  void load_warehouse(int w_id);
  void load_district(int w_id);
  void load_history(int w_id);
  void load_order(int w_id);
  void load_customer(int w_id);

  // cust_utils
  uint64_t cust_derive_key(char *c_last, int c_d_id, int c_w_id);
  int set_last_name(int num, char *name);

  // struct txn::TXN gen_insert_txn(uint64_t key, void *rec) {}
  // struct txn::TXN gen_upd_txn(uint64_t key, void *rec) {}

  // void *gen_txn(int wid) {}

  // void exec_txn(void *stmts) { return; }

  // TODO: clean-up
  ~TPCC() {}
  TPCC(std::string name = "TPCC", int num_warehouses = 1);
};

}  // namespace bench

#endif /* TPCC_HPP_ */
