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

#ifndef BENCH_TPCC_64_HPP_
#define BENCH_TPCC_64_HPP_

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

#include "indexes/hash_index.hpp"
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
#define debug_dont_load_order false

// SIGMOD 20
#define index_on_order_tbl false  // also cascade to orderlne, neworder table

#if diascld40
#define TPCC_MAX_ORD_PER_DIST 200000  // 2 master - 2 socket
#elif diascld48
#define TPCC_MAX_ORD_PER_DIST 350000  // 2 master - 2 socket
#elif icc148
#define TPCC_MAX_ORD_PER_DIST 2000000  // 2 master - 1 socket
#else
#define TPCC_MAX_ORD_PER_DIST 200000
#endif

#define NO_MIX 100
#define P_MIX 0
#define OS_MIX 0
#define D_MIX 0
#define SL_MIX 0  // DONT CHANGE, BROKEN
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
#define TPCC_MAX_ITEMS 100000
#define TPCC_NCUST_PER_DIST 3000
#define TPCC_NDIST_PER_WH 10
#define TPCC_ORD_PER_DIST 3000

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

typedef uint64_t date_t;
constexpr char csv_delim = '|';

class TPCC : public Benchmark {
 private:
  const uint tpch_scale_factor;

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
    uint32_t n_nationkey;
    char n_name[16];  // var
    uint32_t n_regionkey;
    char n_comment[115];  // var
  };

  struct __attribute__((packed)) ch_region {
    uint32_t r_regionkey;
    char r_name[12];      // var
    char r_comment[115];  // var
  };

  struct __attribute__((packed)) ch_supplier {
    uint32_t suppkey;
    char s_name[18];     // fix
    char s_address[41];  // var
    uint32_t s_nationkey;
    char s_phone[15];  // fix
    double s_acctbal;
    char s_comment[101];  // var
  };

  struct __attribute__((packed)) tpcc_stock {
    uint32_t s_i_id;
    uint32_t s_w_id;
    int32_t s_quantity;
    char s_dist[TPCC_NDIST_PER_WH][24];
    uint32_t s_ytd;
    uint32_t s_order_cnt;
    uint32_t s_remote_cnt;
    char s_data[51];
    uint32_t s_su_suppkey;  // ch-specific
  };

  struct __attribute__((packed)) tpcc_item {
    uint32_t i_id;
    uint32_t i_im_id;
    char i_name[25];
    double i_price;
    char i_data[51];
  };

  struct __attribute__((packed)) tpcc_warehouse {
    uint32_t w_id;
    char w_name[11];
    char w_street[2][21];
    char w_city[21];
    char w_state[2];
    char w_zip[9];
    double w_tax;
    double w_ytd;
  };

  struct __attribute__((packed)) tpcc_district {
    uint32_t d_id;
    uint32_t d_w_id;
    char d_name[11];
    char d_street[2][21];
    char d_city[21];
    char d_state[2];
    char d_zip[9];
    double d_tax;
    double d_ytd;
    uint64_t d_next_o_id;
  };
  struct __attribute__((packed)) tpcc_history {
    uint32_t h_c_id;
    uint32_t h_c_d_id;
    uint32_t h_c_w_id;
    uint32_t h_d_id;
    uint32_t h_w_id;
    uint64_t h_date;
    double h_amount;
    char h_data[25];
  };
  struct __attribute__((packed)) tpcc_customer {
    uint32_t c_id;
    uint32_t c_w_id;
    uint32_t c_d_id;
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
    double c_credit_lim;
    double c_discount;
    double c_balance;
    double c_ytd_payment;
    uint32_t c_payment_cnt;
    uint32_t c_delivery_cnt;
    char c_data[501];
    uint32_t c_n_nationkey;
  };

  struct __attribute__((packed)) tpcc_order {
    uint32_t o_id;
    uint32_t o_d_id;
    uint32_t o_w_id;
    uint32_t o_c_id;
    date_t o_entry_d;
    uint32_t o_carrier_id;
    uint32_t o_ol_cnt;
    uint32_t o_all_local;
  };

  struct __attribute__((packed)) tpcc_order_line {
    uint32_t ol_o_id;
    uint32_t ol_d_id;
    uint32_t ol_w_id;
    uint32_t ol_number;
    uint32_t ol_i_id;
    uint32_t ol_supply_w_id;
    date_t ol_delivery_d;
    uint32_t ol_quantity;
    double ol_amount;
    // char ol_dist_info[24]; // TODO: uncomment
  };

  struct __attribute__((packed)) tpcc_order_line_batch {
    uint32_t ol_o_id[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_d_id[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_w_id[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_number[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_i_id[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_supply_w_id[TPCC_MAX_OL_PER_ORDER];
    date_t ol_delivery_d[TPCC_MAX_OL_PER_ORDER];
    uint32_t ol_quantity[TPCC_MAX_OL_PER_ORDER];
    double ol_amount[TPCC_MAX_OL_PER_ORDER];
    // char ol_dist_info[TPCC_MAX_OL_PER_ORDER][24]; // TODO: uncomment
  };

  struct __attribute__((packed)) tpcc_new_order {
    uint32_t no_o_id;
    uint32_t no_d_id;
    uint32_t no_w_id;
  };

  struct secondary_record {
    int sr_idx;
    int sr_nids;
    uint32_t *sr_rids;
#define NDEFAULT_RIDS 16
  };

  struct cust_read {
    uint32_t c_id;
    uint32_t c_d_id;
    uint32_t c_w_id;
    char c_first[FIRST_NAME_LEN + 1];
    char c_last[LAST_NAME_LEN + 1];
  };

  struct item {
    uint32_t ol_i_id;
    uint32_t ol_supply_w_id;
    uint32_t ol_quantity;
  };

  // neworder tpcc query
  struct tpcc_query {
    TPCC_QUERY_TYPE query_type;
    uint32_t w_id;
    uint32_t d_id;
    uint32_t c_id;
    int threshold;
    uint32_t o_carrier_id;
    uint32_t d_w_id;
    uint32_t c_w_id;
    uint32_t c_d_id;
    char c_last[LAST_NAME_LEN];
    double h_amount;
    uint8_t by_last_name;
    struct item item[TPCC_MAX_OL_PER_ORDER];
    char rbk;
    char remote;
    uint32_t ol_cnt;
    date_t o_entry_d;
  };

  // shortcut for secondary index
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

  void load_supplier(int w_id, uint64_t xid, ushort partition_id,
                     ushort master_ver);
  void load_nation(int w_id, uint64_t xid, ushort partition_id,
                   ushort master_ver);
  void load_region(int w_id, uint64_t xid, ushort partition_id,
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
    return storage::MemoryManager::alloc(
        sizeof(struct tpcc_query),
        storage::NUMAPartitionPolicy::getInstance()
            .getPartitionInfo(pid)
            .numa_idx,
        MADV_DONTFORK);
  }
  void free_query_struct_ptr(void *ptr) {
    storage::MemoryManager::free(ptr);  //, sizeof(struct tpcc_query));
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
       uint tpch_scale_factor = 0, int g_dist_threshold = 0,
       std::string csv_path = "", bool is_ch_benchmark = false);

  static_assert(!(D_MIX > 0 && !index_on_order_tbl),
                "Delivery Txn requires index on order tables");

  friend std::ostream &operator<<(std::ostream &out, const TPCC::ch_nation &r);
  friend std::ostream &operator<<(std::ostream &out, const TPCC::ch_region &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::ch_supplier &r);
  friend std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_stock &r);
  friend std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_item &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::tpcc_warehouse &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::tpcc_district &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::tpcc_history &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::tpcc_customer &r);
  friend std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_order &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::tpcc_order_line &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::tpcc_new_order &r);
};

std::ostream &operator<<(std::ostream &out, const TPCC::ch_nation &r);
std::ostream &operator<<(std::ostream &out, const TPCC::ch_region &r);
std::ostream &operator<<(std::ostream &out, const TPCC::ch_supplier &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_stock &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_item &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_warehouse &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_district &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_history &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_customer &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_order &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_order_line &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_new_order &r);

}  // namespace bench

#endif /* BENCH_TPCC_64_HPP_ */
