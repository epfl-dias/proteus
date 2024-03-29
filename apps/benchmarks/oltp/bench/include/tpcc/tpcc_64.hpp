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
#include <olap/plan/prepared-statement.hpp>
#include <oltp/common/common.hpp>
#include <oltp/common/constants.hpp>
#include <oltp/common/numa-partition-policy.hpp>
#include <oltp/interface/bench.hpp>
#include <oltp/storage/table.hpp>
#include <oltp/transaction/transaction_manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/util/erase-constructor-idioms.hpp>
#include <random>
#include <string>
#include <thread>

#include "constants.hpp"

namespace bench {

/*
  Benchmark: TPC-C
  Spec: http://www.tpc.org/tpc_documents_current_versions/pdf/tpc-c_v5.11.0.pdf
*/

class TpccTxnGen;

enum TPCC_QUERY_TYPE {
  NEW_ORDER,
  PAYMENT,
  ORDER_STATUS,
  DELIVERY,
  STOCK_LEVEL
};

class TPCC : public Benchmark, proteus::utils::remove_copy_move {
 private:
  const uint tpch_scale_factor;

  storage::Schema *schema;
  std::shared_ptr<storage::Table> table_warehouse{};
  std::shared_ptr<storage::Table> table_district{};
  std::shared_ptr<storage::Table> table_customer{};
  std::shared_ptr<storage::Table> table_history{};
  std::shared_ptr<storage::Table> table_new_order{};
  std::shared_ptr<storage::Table> table_order{};
  std::shared_ptr<storage::Table> table_order_line{};
  std::shared_ptr<storage::Table> table_stock{};
  std::shared_ptr<storage::Table> table_item{};

  std::shared_ptr<storage::Table> table_region{};
  std::shared_ptr<storage::Table> table_nation{};
  std::shared_ptr<storage::Table> table_supplier{};

  int num_warehouse;
  int g_dist_threshold;
  unsigned int seed;
  // TPCC_QUERY_TYPE sequence[MIX_COUNT]{};
  std::vector<TPCC_QUERY_TYPE> query_sequence;
  std::string csv_path;
  const bool is_ch_benchmark;

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
    uint32_t d_next_o_id;
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

  struct __attribute__((packed)) tpcc_orderline {
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

  struct __attribute__((packed)) tpcc_orderline_batch {
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

  void load_stock(int w_id, xid_t xid, partition_id_t partition_id,
                  master_version_t master_ver);
  void load_item(int w_id, xid_t xid, partition_id_t partition_id,
                 master_version_t master_ver);
  void load_warehouse(int w_id, xid_t xid, partition_id_t partition_id,
                      master_version_t master_ver);
  void load_district(int w_id, xid_t xid, partition_id_t partition_id,
                     master_version_t master_ver);
  void load_history(int w_id, xid_t xid, partition_id_t partition_id,
                    master_version_t master_ver);
  void load_order(int w_id, xid_t xid, partition_id_t partition_id,
                  master_version_t master_ver);
  void load_customer(int w_id, xid_t xid, partition_id_t partition_id,
                     master_version_t master_ver);

  void load_supplier(int w_id, xid_t xid, partition_id_t partition_id,
                     master_version_t master_ver);
  void load_nation(int w_id, xid_t xid, partition_id_t partition_id,
                   master_version_t master_ver);
  void load_region(int w_id, xid_t xid, partition_id_t partition_id,
                   master_version_t master_ver);

  void pre_run(worker_id_t wid, xid_t xid, partition_id_t partition_id,
               master_version_t master_ver);
  void post_run(worker_id_t wid, xid_t xid, partition_id_t partition_id,
                master_version_t master_ver) {
    if (wid == 0) {
      // consistency check parallelize itself, doesnt need a worker to do it.
      // moreover, it is not performance-critical.
      verify_consistency();
    }
  }

  // consistency checks
  void verify_consistency();
  bool consistency_check_1(bool print_inconsistent_rows = true);
  bool consistency_check_2(bool print_inconsistent_rows = true);
  bool consistency_check_3(bool print_inconsistent_rows = true);
  bool consistency_check_4(bool print_inconsistent_rows = true);

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

  static inline date_t __attribute__((always_inline)) get_timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
  }

  // cust_utils
  static size_t cust_derive_key(const char *c_last, uint32_t c_d_id,
                                uint32_t c_w_id);
  static auto set_last_name(size_t num, char *name);
  uint fetch_cust_records(const struct secondary_record &sr,
                          struct cust_read *c_recs, const txn::TxnTs &txnTs);

  // get queries
  void tpcc_get_next_payment_query(int wid, void *arg) const;
  void tpcc_get_next_neworder_query(int wid, void *arg) const;
  void tpcc_get_next_orderstatus_query(int wid, void *arg) const;
  void tpcc_get_next_delivery_query(int wid, void *arg) const;
  void tpcc_get_next_stocklevel_query(int wid, void *arg) const;

  bool exec_micro_scan_stock(txn::Txn &txn);

  bool exec_neworder_txn(const struct tpcc_query *stmts, txn::Txn &txn);
  bool exec_payment_txn(const struct tpcc_query *stmts, txn::Txn &txn);
  bool exec_orderstatus_txn(const struct tpcc_query *stmts, txn::Txn &txn);
  bool exec_delivery_txn(const struct tpcc_query *stmts, txn::Txn &txn);
  bool exec_stocklevel_txn(const struct tpcc_query *stmts, txn::Txn &txn);

  bool exec_txn(txn::TransactionExecutor &executor, txn::Txn &txn,
                void *params);

  void gen_txn(worker_id_t wid, void *txn_ptr, partition_id_t partition_id);
  [[maybe_unused]] static void print_tpcc_query(void *arg);

  BenchQueue *getBenchQueue(worker_id_t workerId,
                            partition_id_t partitionId) override {
    auto *benchQueuePtr =
        MemoryManager::mallocPinned(sizeof(TpccTxnGen) + alignof(TpccTxnGen));
    auto *benchQueue =
        new (benchQueuePtr) TpccTxnGen(*this, workerId, partitionId);

    return benchQueue;
    //    return dynamic_cast<BenchQueue *>(
    //        new TpccTxnGen(*this, workerId, partitionId));
  }

  void clearBenchQueue(BenchQueue *pt) override {
    pt->~BenchQueue();
    MemoryManager::freePinned(pt);
  }

  ~TPCC() override;
  explicit TPCC(std::string name = "TPCC", int num_warehouses = 1,
                int active_warehouse = 1,
                const std::vector<TPCC_QUERY_TYPE> &query_seq = {},
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
                                  const TPCC::tpcc_orderline &r);
  friend std::ostream &operator<<(std::ostream &out,
                                  const TPCC::tpcc_new_order &r);

 public:
  PreparedStatement consistency_check_1_query_builder(
      bool return_aggregate = true, string olap_plugin = "block-remote");
  PreparedStatement consistency_check_1_query_builder_1t(
      bool return_aggregate = true, string olap_plugin = "block-remote");
  std::vector<PreparedStatement> consistency_check_2_query_builder(
      bool return_aggregate = true, string olap_plugin = "block-remote");
  PreparedStatement consistency_check_3_query_builder(
      bool return_aggregate = true, string olap_plugin = "block-remote");
  PreparedStatement consistency_check_4_query_builder(
      bool return_aggregate = true, string olap_plugin = "block-remote");

 private:
  static inline void shuffle_sequence(std::vector<TPCC_QUERY_TYPE> &q_seq) {
    static std::random_device rd;
    static std::mt19937 g(rd());
    std::shuffle(q_seq.begin(), q_seq.end(), g);
  }

 public:
  //-- generator
  class TpccTxnGen : public bench::BenchQueue {
    TpccTxnGen(TPCC &TpccTxnGen, worker_id_t wid, partition_id_t partition_id)
        : tpccBench(TpccTxnGen), wid(wid), partition_id(partition_id) {
      this->_txn_mem =
          static_cast<struct tpcc_query *>(MemoryManager::mallocPinnedOnNode(
              sizeof(struct tpcc_query),
              storage::NUMAPartitionPolicy::getInstance()
                  .getPartitionInfo(partition_id)
                  .numa_idx));
    }
    ~TpccTxnGen() override { MemoryManager::freePinned(_txn_mem); }

    txn::StoredProcedure pop(worker_id_t workerId,
                             partition_id_t partitionId) override {
      if (tpccBench.num_readonly_worker > workerId) {
        // micro-bench: execute a read-only scan-aggregation
        return txn::StoredProcedure(
            [&](txn::TransactionExecutor &executor, txn::Txn &txn,
                void *params) { return tpccBench.exec_micro_scan_stock(txn); },
            this->_txn_mem, true);
      } else {
        // generate and execute a TPCC-txn
        tpccBench.gen_txn(this->wid, this->_txn_mem, this->partition_id);
        return txn::StoredProcedure(
            [&](txn::TransactionExecutor &executor, txn::Txn &txn,
                void *params) {
              return tpccBench.exec_txn(executor, txn, params);
            },
            this->_txn_mem);
      }
    }

    void pre_run() override { tpccBench.pre_run(wid, 0, partition_id, 0); }
    void post_run() override {
      tpccBench.post_run(
          wid, std::numeric_limits<xid_t>::max(), partition_id,
          txn::TransactionManager::getInstance().get_current_master_version());
    }
    void dump(std::string name) override {}

   private:
    struct tpcc_query *_txn_mem;
    TPCC &tpccBench;
    const worker_id_t wid;
    const partition_id_t partition_id;

    friend class TPCC;
  };
  //-- END generator

  friend class TpccTxnGen;
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
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_orderline &r);
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_new_order &r);

}  // namespace bench

#endif /* BENCH_TPCC_64_HPP_ */
