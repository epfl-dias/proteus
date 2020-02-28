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

#ifndef BENCH_TPCC_HPP_
#define BENCH_TPCC_HPP_

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "interfaces/bench.hpp"
#include "scheduler/topology.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"

// FIXME: REPLICATED_ITEM_TABLE - broken, incomplete stuff.
#define REPLICATED_ITEM_TABLE false

#define PARTITION_LOCAL_ITEM_TABLE true
#define tpcc_dist_txns false
#define tpcc_cust_sec_idx false
#define batch_insert_no_ol true

#define NO_MIX 95
#define P_MIX 0
#define OS_MIX 0
#define D_MIX 5
#define SL_MIX 0
#define MIX_COUNT 100

// #define NO_MIX 45
// #define P_MIX 43
// #define OS_MIX 4
// #define D_MIX 4
// #define SL_MIX 4
// #define MIX_COUNT 100

#define FIRST_NAME_MIN_LEN 8
#define FIRST_NAME_LEN 16
#define LAST_NAME_LEN 16
#define TPCC_MAX_OL_PER_ORDER 15
#define MAX_OPS_PER_QUERY 255

// From TPCC-SPEC
#define TPCC_MAX_ITEMS 100000  // 100000
#define TPCC_NCUST_PER_DIST 3000
#define TPCC_NDIST_PER_WH 10
#define TPCC_ORD_PER_DIST 3000

#define TPCC_MAX_ORD_PER_DIST 250000
#define TPCC_MAX_ORDER_INITIAL_CAP_PER_PARTITION \
  (TPCC_MAX_ORD_PER_DIST * TPCC_NDIST_PER_WH * NUM_CORE_PER_SOCKET)
#define TPCC_MAX_ORDER_INITIAL_CAP \
  (TPCC_MAX_ORDER_INITIAL_CAP_PER_PARTITION * NUM_SOCKETS)

#define MAKE_STOCK_KEY(w, s) (w * TPCC_MAX_ITEMS + s)
#define MAKE_DIST_KEY(w, d) (w * TPCC_NDIST_PER_WH + d)
#define MAKE_CUST_KEY(w, d, c) (MAKE_DIST_KEY(w, d) * TPCC_NCUST_PER_DIST + c)

#define MAKE_ORDER_KEY(w, d, o) \
  ((MAKE_DIST_KEY(w, d) * TPCC_MAX_ORD_PER_DIST) + o)
#define MAKE_OL_KEY(w, d, o, ol) \
  (MAKE_ORDER_KEY(w, d, o) * TPCC_MAX_OL_PER_ORDER + ol)

namespace bench {

/*
  Benchmark: TPC-C
  Spec: http://www.tpc.org/tpc_documents_current_versions/pdf/tpc-c_v5.11.0.pdf
*/

enum TPCC_QUERY_TYPE {
  NEW_ORDER,
  PAYMENT,
  ORDER_STATUS,
  DELIVERY,
  STOCK_LEVEL
};

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
  storage::Table *table_stock;

#if REPLICATED_ITEM_TABLE
  storage::Table *table_item[NUM_SOCKETS];
#else
  storage::Table *table_item;
#endif

  storage::Table *table_region;
  storage::Table *table_nation;
  storage::Table *table_supplier;

  int num_warehouse;
  int g_dist_threshold;
  unsigned int seed;
  TPCC_QUERY_TYPE sequence[MIX_COUNT];
  std::string csv_path;
  const bool is_ch_benchmark;
  const bool layout_column_store;

 public:
  struct __attribute__((packed)) ch_nation {
    ushort n_nationkey;
    char n_name[16];  // var
    ushort n_regionkey;
    char n_comment[115];  // var
  };

  struct __attribute__((packed)) ch_region {
    ushort r_regionkey;
    char r_name[12];      // var
    char r_comment[115];  // var
  };

  struct __attribute__((packed)) ch_supplier {
    uint32_t suppkey;
    char s_name[18];     // fix
    char s_address[41];  // var
    ushort s_nationkey;
    char s_phone[15];  // fix
    float s_acctbal;
    char s_comment[101];  // var
  };

  struct __attribute__((packed)) tpcc_stock {
    uint32_t s_i_id;
    ushort s_w_id;
    short s_quantity;
    char s_dist[TPCC_NDIST_PER_WH][24];
    ushort s_ytd;
    ushort s_order_cnt;
    ushort s_remote_cnt;
    char s_data[51];
    uint32_t s_su_suppkey;  // ch-specific
  };

  struct __attribute__((packed)) tpcc_item {
    uint32_t i_id;
    uint32_t i_im_id;
    char i_name[25];
    float i_price;
    char i_data[51];
  };

  struct __attribute__((packed)) tpcc_warehouse {
    ushort w_id;
    char w_name[11];
    char w_street[2][21];
    char w_city[21];
    char w_state[2];
    char w_zip[9];
    float w_tax;
    float w_ytd;
  };

  struct __attribute__((packed)) tpcc_district {
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
  struct __attribute__((packed)) tpcc_history {
    uint32_t h_c_id;
    ushort h_c_d_id;
    ushort h_c_w_id;
    ushort h_d_id;
    ushort h_w_id;
    uint64_t h_date;
    float h_amount;
    char h_data[25];
  };
  struct __attribute__((packed)) tpcc_customer {
    uint32_t c_id;
    ushort c_w_id;
    ushort c_d_id;
    char c_first[FIRST_NAME_LEN + 1];
    char c_middle[2];
    char c_last[LAST_NAME_LEN + 1];
    char c_street[2][21];
    char c_city[21];
    char c_state[2];
    char c_zip[9];
    char c_phone[16];
    uint64_t c_since;
    char c_credit[2];
    float c_credit_lim;
    float c_discount;
    float c_balance;
    float c_ytd_payment;
    ushort c_payment_cnt;
    ushort c_delivery_cnt;
    char c_data[501];
    ushort c_n_nationkey;
  };

  struct __attribute__((packed)) tpcc_order {
    uint64_t o_id;
    ushort o_d_id;
    ushort o_w_id;
    uint32_t o_c_id;
    uint32_t o_entry_d;  // TODO: MAKE IT 64 BIT
    short o_carrier_id;
    ushort o_ol_cnt;
    ushort o_all_local;
  };

  struct __attribute__((packed)) tpcc_order_line {
    uint64_t ol_o_id;
    ushort ol_d_id;
    ushort ol_w_id;
    ushort ol_number;
    uint32_t ol_i_id;
    ushort ol_supply_w_id;
    uint32_t ol_delivery_d;  // TODO: MAKE IT 64 BIT
    ushort ol_quantity;
    float ol_amount;
    // char ol_dist_info[24]; // TODO: uncomment
  };

  struct __attribute__((packed)) tpcc_order_line_batch {
    uint64_t ol_o_id[TPCC_MAX_OL_PER_ORDER];
    ushort ol_d_id[TPCC_MAX_OL_PER_ORDER];
    ushort ol_w_id[TPCC_MAX_OL_PER_ORDER];
    ushort ol_number[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_i_id[TPCC_MAX_OL_PER_ORDER];
    ushort ol_supply_w_id[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_delivery_d[TPCC_MAX_OL_PER_ORDER];  // TODO: MAKE IT 64 BIT
    ushort ol_quantity[TPCC_MAX_OL_PER_ORDER];
    float ol_amount[TPCC_MAX_OL_PER_ORDER];
    // char ol_dist_info[TPCC_MAX_OL_PER_ORDER][24]; // TODO: uncomment
  };

  struct __attribute__((packed)) tpcc_new_order {
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

  struct cust_read {
    uint32_t c_id;
    ushort c_d_id;
    ushort c_w_id;
    char c_first[FIRST_NAME_LEN + 1];
    char c_last[LAST_NAME_LEN + 1];
  };

  struct item {
    uint32_t ol_i_id;
    ushort ol_supply_w_id;
    ushort ol_quantity;
  };

  // neworder tpcc query
  struct tpcc_query {
    TPCC_QUERY_TYPE query_type;
    ushort w_id;
    ushort d_id;
    uint32_t c_id;
    int threshold;
    int o_carrier_id;
    ushort d_w_id;
    ushort c_w_id;
    ushort c_d_id;
    char c_last[LAST_NAME_LEN];
    float h_amount;
    uint8_t by_last_name;
    struct item item[TPCC_MAX_OL_PER_ORDER];
    char rbk;
    char remote;
    ushort ol_cnt;
    uint32_t o_entry_d;  // 64bit please
  };

  // fucking shortcut
  indexes::HashIndex<uint64_t, struct secondary_record> *cust_sec_index;

  void init_tpcc_seq_array();
  void create_tbl_warehouse(uint64_t num_warehouses);
  void create_tbl_district(uint64_t num_districts);
  void create_tbl_customer(uint64_t num_cust);
  void create_tbl_history(uint64_t num_history);
  void create_tbl_new_order(uint64_t num_new_order);
  void create_tbl_order(uint64_t num_order);
  void create_tbl_order_line(uint64_t num_order_line);
  void create_tbl_item(uint64_t num_item);
  void create_tbl_stock(uint64_t num_stock);

  // ch-tables
  void create_tbl_supplier(uint64_t num_supp);
  void create_tbl_region(uint64_t num_region);
  void create_tbl_nation(uint64_t num_nation);

  void load_data(int num_threads = 1);
  void load_stock(int w_id, uint64_t xid, ushort partition_id,
                  ushort master_ver);
  void load_item(int w_id, uint64_t xid, ushort partition_id,
                 ushort master_ver);
  void load_warehouse(int w_id, uint64_t xid, ushort partition_id,
                      ushort master_ver);
  void load_district(int w_id, uint64_t xid, ushort partition_id,
                     ushort master_ver);
  void load_history(int w_id, uint64_t xid, ushort partition_id,
                    ushort master_ver);
  void load_order(int w_id, uint64_t xid, ushort partition_id,
                  ushort master_ver);
  void load_customer(int w_id, uint64_t xid, ushort partition_id,
                     ushort master_ver);

  void pre_run(int wid, uint64_t xid, ushort partition_id, ushort master_ver);
  void post_run(int wid, uint64_t xid, ushort partition_id, ushort master_ver) {
    if (wid == 0) {
      table_order->reportUsage();
      table_new_order->reportUsage();
      table_order_line->reportUsage();
    }

    // TODO: Implement verify consistency after txn run.
  }

  // CSV Loaders

  void load_stock_csv(std::string filename = "stock.tbl", char delim = '|');
  void load_item_csv(std::string filename = "item.tbl", char delim = '|');
  void load_warehouse_csv(std::string filename = "warehouse.tbl",
                          char delim = '|');
  void load_district_csv(std::string filename = "district.tbl",
                         char delim = '|');
  void load_history_csv(std::string filename = "history.tbl", char delim = '|');
  void load_order_csv(std::string filename = "order.tbl", char delim = '|');
  void load_customer_csv(std::string filename = "customer.tbl",
                         char delim = '|');
  void load_nation_csv(std::string filename = "nation.tbl", char delim = '|');
  void load_neworder_csv(std::string filename = "neworder.tbl",
                         char delim = '|');
  void load_orderline_csv(std::string filename = "orderline.tbl",
                          char delim = '|');
  void load_region_csv(std::string filename = "region.tbl", char delim = '|');
  void load_supplier_csv(std::string filename = "supplier.tbl",
                         char delim = '|');
  void load_customer_secondary_index(struct tpcc_customer &r);

  void *get_query_struct_ptr(ushort pid) {
    return storage::MemoryManager::alloc(sizeof(struct tpcc_query), pid,
                                         MADV_DONTFORK);
  }
  void free_query_struct_ptr(void *ptr) {
    storage::MemoryManager::free(ptr, sizeof(struct tpcc_query));
  }

  // cust_utils
  uint64_t cust_derive_key(char *c_last, int c_d_id, int c_w_id);
  int set_last_name(int num, char *name);
  uint fetch_cust_records(const struct secondary_record &sr,
                          struct cust_read *c_recs, uint64_t xid);

  // get queries
  void tpcc_get_next_payment_query(int wid, void *arg);
  void tpcc_get_next_neworder_query(int wid, void *arg);
  void tpcc_get_next_orderstatus_query(int wid, void *arg);
  void tpcc_get_next_delivery_query(int wid, void *arg);
  void tpcc_get_next_stocklevel_query(int wid, void *arg);

  bool exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                ushort delta_ver, ushort partition_id);
  void gen_txn(int wid, void *txn_ptr, ushort partition_id);

  bool exec_neworder_txn(const struct tpcc_query *stmts, uint64_t xid,
                         ushort master_ver, ushort delta_ver,
                         ushort partition_id);
  bool exec_payment_txn(struct tpcc_query *stmts, uint64_t xid,
                        ushort master_ver, ushort delta_ver,
                        ushort partition_id);
  bool exec_orderstatus_txn(struct tpcc_query *stmts, uint64_t xid,
                            ushort master_ver, ushort delta_ver,
                            ushort partition_id);
  bool exec_delivery_txn(struct tpcc_query *stmts, uint64_t xid,
                         ushort master_ver, ushort delta_ver,
                         ushort partition_id);
  bool exec_stocklevel_txn(struct tpcc_query *stmts, uint64_t xid,
                           ushort master_ver, ushort delta_ver,
                           ushort partition_id);
  void print_tpcc_query(void *arg);

  ~TPCC() {}
  TPCC(std::string name = "TPCC", int num_warehouses = 1,
       int active_warehouse = 1, bool layout_column_store = true,
       int g_dist_threshold = 0, std::string csv_path = "",
       bool is_ch_benchmark = false);
};

}  // namespace bench

#endif /* BENCH_TPCC_HPP_ */
