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

#include "tpcc/tpcc_64.hpp"

#include <chrono>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <oltp/common/utils.hpp>
#include <platform/threadpool/thread.hpp>
#include <utility>

#define NDEFAULT_RIDS 16

namespace bench {

void TPCC::init_tpcc_seq_array() {
  this->query_sequence.reserve(MIX_COUNT);
  int total = 0;
  for (int i = 0; i < NO_MIX; ++i) {
    query_sequence.emplace_back(NEW_ORDER);
  }
  total = NO_MIX;
  for (int i = 0; i < P_MIX; ++i) {
    query_sequence.emplace_back(PAYMENT);
  }
  total = total + P_MIX;
  for (int i = 0; i < OS_MIX; ++i) {
    query_sequence.emplace_back(ORDER_STATUS);
  }
  total = total + OS_MIX;
  for (int i = 0; i < D_MIX; ++i) {
    query_sequence.emplace_back(DELIVERY);
  }
  total = total + D_MIX;
  for (int i = 0; i < SL_MIX; ++i) {
    query_sequence.emplace_back(STOCK_LEVEL);
  }
  total = total + SL_MIX;
  //  // shuffle elements of the sequence array
  //  srand(time(nullptr));
  //  for (int i = MIX_COUNT - 1; i > 0; i--) {
  //    int j = rand() % (i + 1);
  //    TPCC_QUERY_TYPE temp = sequence[i];
  //    sequence[i] = sequence[j];
  //    sequence[j] = temp;
  //  }
}

TPCC::~TPCC() {
  // clear references from shared ptrs
  table_warehouse.reset();
  table_district.reset();
  table_customer.reset();
  table_history.reset();
  table_new_order.reset();
  table_order.reset();
  table_order_line.reset();
  table_stock.reset();
  table_item.reset();
  table_region.reset();
  table_nation.reset();
  table_supplier.reset();

  if (schema->getTable("tpcc_warehouse") != nullptr)
    schema->drop_table("tpcc_warehouse");

  if (schema->getTable("tpcc_district") != nullptr)
    schema->drop_table("tpcc_district");

  if (schema->getTable("tpcc_customer") != nullptr)
    schema->drop_table("tpcc_customer");

  if (schema->getTable("tpcc_history") != nullptr)
    schema->drop_table("tpcc_history");

  if (schema->getTable("tpcc_neworder") != nullptr)
    schema->drop_table("tpcc_neworder");

  if (schema->getTable("tpcc_order") != nullptr)
    schema->drop_table("tpcc_order");

  if (schema->getTable("tpcc_orderline") != nullptr)
    schema->drop_table("tpcc_orderline");

  if (schema->getTable("tpcc_stock") != nullptr)
    schema->drop_table("tpcc_stock");

  if (schema->getTable("tpcc_item") != nullptr) schema->drop_table("tpcc_item");

  if (schema->getTable("tpcc_region") != nullptr)
    schema->drop_table("tpcc_region");

  if (schema->getTable("tpcc_nation") != nullptr)
    schema->drop_table("tpcc_nation");

  if (schema->getTable("tpcc_supplier") != nullptr)
    schema->drop_table("tpcc_supplier");
}

void TPCC::print_tpcc_query(void *arg) {
  struct tpcc_query *q = (struct tpcc_query *)arg;
  std::cout << "-------TPCC QUERY------" << std::endl;
  switch (q->query_type) {
    case NEW_ORDER:
      std::cout << "\tType: NEW_ORDER" << std::endl;
      for (int i = 0; i < q->ol_cnt; i++) {
        std::cout << i << " - " << q->item[i].ol_i_id << std::endl;
      }

      break;
    case PAYMENT:
      std::cout << "\tType: PAYMENT" << std::endl;
      break;
    case ORDER_STATUS:
      std::cout << "\tType: ORDER_STATUS" << std::endl;
      break;
    case DELIVERY:
      std::cout << "\tType: DELIVERY" << std::endl;
      break;
    case STOCK_LEVEL:
      std::cout << "\tType: STOCK_LEVEL" << std::endl;
      break;
    default:
      break;
  }
  std::cout << "\tw_id: " << q->w_id << std::endl;
  std::cout << "\td_id: " << q->d_id << std::endl;
  std::cout << "\tc_id: " << q->c_id << std::endl;
  std::cout << "\tol_cnt: " << q->ol_cnt << std::endl;

  std::cout << "-----------------------" << std::endl;
}

TPCC::TPCC(std::string name, int num_warehouses, int active_warehouse,
           bool layout_column_store, std::vector<TPCC_QUERY_TYPE> query_seq,
           uint tpch_scale_factor, int g_dist_threshold, std::string csv_path,
           bool is_ch_benchmark)
    : Benchmark(std::move(name), active_warehouse,
                proteus::thread::hardware_concurrency(), g_num_partitions),
      num_warehouse(num_warehouses),
      g_dist_threshold(g_dist_threshold),
      csv_path(std::move(csv_path)),
      is_ch_benchmark(is_ch_benchmark),
      layout_column_store(layout_column_store),
      tpch_scale_factor(tpch_scale_factor) {
  this->schema = &storage::Schema::getInstance();
  this->seed = rand();

  uint64_t total_districts = TPCC_NDIST_PER_WH * (this->num_warehouse);
  uint64_t max_customers = TPCC_NCUST_PER_DIST * total_districts;
  uint64_t max_orders = TPCC_MAX_ORD_PER_DIST * total_districts;

  // max_orders = max_orders * g_num_partitions;
  // std::cout << "MAX ORDERS: " << max_orders << std::endl;
  // std::cout << "g_num_partitions: " << g_num_partitions << std::endl;
  // uint64_t max_order_line = TPCC_MAX_OL_PER_ORDER * max_orders;

  uint64_t max_order_line = TPCC_MAX_OL_PER_ORDER * max_orders;
  uint64_t max_stock = TPCC_MAX_ITEMS * (this->num_warehouse);

  std::vector<proteus::thread> loaders;

  loaders.emplace_back(
      [this]() { this->create_tbl_warehouse(this->num_warehouse); });
  loaders.emplace_back([this, total_districts]() {
    this->create_tbl_district(total_districts);
  });
  loaders.emplace_back(
      [this, max_customers]() { this->create_tbl_customer(max_customers); });
  loaders.emplace_back([this, max_orders, max_customers]() {
    if (P_MIX > 0)
      this->create_tbl_history(max_orders / 2);
    else
      this->create_tbl_history(max_customers);
  });
  loaders.emplace_back([this]() { this->create_tbl_item(TPCC_MAX_ITEMS); });
  loaders.emplace_back(
      [this, max_stock]() { this->create_tbl_stock(max_stock); });

#if !debug_dont_load_order
  loaders.emplace_back(
      [this, max_orders]() { this->create_tbl_new_order(max_orders); });
  loaders.emplace_back(
      [this, max_orders]() { this->create_tbl_order(max_orders); });

  loaders.emplace_back([this, max_order_line]() {
    this->create_tbl_order_line(max_order_line);
  });
#endif

  // this->create_tbl_warehouse(this->num_warehouse);

  // this->create_tbl_district(total_districts);

  // this->create_tbl_customer(max_customers);

  // this->create_tbl_history(max_customers);

  // this->create_tbl_new_order(max_orders);

  // this->create_tbl_order(max_orders);

  // this->create_tbl_order_line(max_order_line);

  // this->create_tbl_item(TPCC_MAX_ITEMS);

  // this->create_tbl_stock(max_stock);

  for (auto &th : loaders) {
    th.join();
  }

  if (is_ch_benchmark) {
    this->create_tbl_supplier(10000);
    this->create_tbl_nation(150);
    this->create_tbl_region(5);
  }

  cust_sec_index = new indexes::HashIndex<uint64_t, struct secondary_record>();
  // cust_sec_index->reserve(max_customers);

  this->schema->memoryReport();

  if (query_seq.empty()) {
    init_tpcc_seq_array();
  } else {
    this->query_sequence = query_seq;
  }
}

void TPCC::create_tbl_warehouse(uint64_t num_warehouses) {
  // Primary Key: W_ID
  storage::TableDef columns;

  struct tpcc_warehouse tmp {};

  columns.emplace_back("w_id", storage::INTEGER, sizeof(tmp.w_id));

  columns.emplace_back("w_name", storage::VARCHAR,
                       sizeof(tmp.w_name));  // size 10 +1 for null character

  columns.emplace_back("w_street_1", storage::VARCHAR, sizeof(tmp.w_street[0]));
  columns.emplace_back("w_street_2", storage::VARCHAR, sizeof(tmp.w_street[0]));
  columns.emplace_back("w_city", storage::VARCHAR, sizeof(tmp.w_city));
  columns.emplace_back("w_state", storage::STRING, sizeof(tmp.w_state));
  columns.emplace_back("w_zip", storage::STRING, sizeof(tmp.w_zip));
  columns.emplace_back("w_tax", storage::FLOAT, sizeof(tmp.w_tax));
  columns.emplace_back("w_ytd", storage::FLOAT, sizeof(tmp.w_ytd));

  LOG(INFO) << "Number of warehouses: " << num_warehouses;

  table_warehouse = schema->create_table(
      "tpcc_warehouse",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_warehouses);
}

void TPCC::create_tbl_district(uint64_t num_districts) {
  // Primary Key: (D_W_ID, D_ID) D_W_ID
  // Foreign Key, references W_ID
  storage::TableDef columns;

  struct tpcc_district tmp {};

  columns.emplace_back("d_id", storage::INTEGER, sizeof(tmp.d_id));
  columns.emplace_back("d_w_id", storage::INTEGER, sizeof(tmp.d_w_id));
  columns.emplace_back("d_name", storage::VARCHAR, sizeof(tmp.d_name));
  columns.emplace_back("d_street_1", storage::VARCHAR, sizeof(tmp.d_street[0]));
  columns.emplace_back("d_street_2", storage::VARCHAR, sizeof(tmp.d_street[1]));
  columns.emplace_back("d_city", storage::VARCHAR, sizeof(tmp.d_city));
  columns.emplace_back("d_state", storage::STRING, sizeof(tmp.d_state));
  columns.emplace_back("d_zip", storage::STRING, sizeof(tmp.d_zip));
  columns.emplace_back("d_tax", storage::FLOAT, sizeof(tmp.d_tax));
  columns.emplace_back("d_ytd", storage::FLOAT, sizeof(tmp.d_ytd));
  columns.emplace_back("d_next_o_id", storage::INTEGER,
                       sizeof(tmp.d_next_o_id));

  table_district = schema->create_table(
      "tpcc_district",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_districts);
}

void TPCC::create_tbl_item(uint64_t num_item) {
  // Primary Key: I_ID
  storage::TableDef columns;

  struct tpcc_item tmp;

  columns.emplace_back("i_id", storage::INTEGER, sizeof(tmp.i_id));
  columns.emplace_back("i_im_id", storage::INTEGER, sizeof(tmp.i_im_id));
  columns.emplace_back("i_name", storage::VARCHAR, sizeof(tmp.i_name));
  columns.emplace_back("i_price", storage::FLOAT, sizeof(tmp.i_price));
  columns.emplace_back("i_data", storage::VARCHAR, sizeof(tmp.i_data));

  table_item = schema->create_table(
      "tpcc_item",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_item);
}

void TPCC::create_tbl_stock(uint64_t num_stock) {
  // Primary Key: (S_W_ID, S_I_ID)
  // S_W_ID Foreign Key, references W_ID
  // S_I_ID Foreign Key, references I_ID
  storage::TableDef columns;

  struct tpcc_stock tmp;

  columns.emplace_back("s_i_id", storage::INTEGER, sizeof(tmp.s_i_id));
  columns.emplace_back("s_w_id", storage::INTEGER, sizeof(tmp.s_w_id));
  columns.emplace_back("s_quantity", storage::INTEGER, sizeof(tmp.s_quantity));

  columns.emplace_back("s_dist_01", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_02", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_03", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_04", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_05", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_06", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_07", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_08", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_09", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_dist_10", storage::STRING, sizeof(tmp.s_dist[0]));
  columns.emplace_back("s_ytd", storage::INTEGER, sizeof(tmp.s_ytd));
  columns.emplace_back("s_order_cnt", storage::INTEGER,
                       sizeof(tmp.s_order_cnt));
  columns.emplace_back("s_remote_cnt", storage::INTEGER,
                       sizeof(tmp.s_remote_cnt));
  columns.emplace_back("s_data", storage::VARCHAR, sizeof(tmp.s_data));
  columns.emplace_back("s_su_suppkey", storage::INTEGER,
                       sizeof(tmp.s_su_suppkey));

  table_stock = schema->create_table(
      "tpcc_stock",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_stock);
}  // namespace bench

void TPCC::create_tbl_history(uint64_t num_history) {
  // Primary Key: none
  // (H_C_W_ID, H_C_D_ID, H_C_ID) Foreign Key, references (C_W_ID, C_D_ID,
  // C_ID)
  // (H_W_ID, H_D_ID) Foreign Key, references (D_W_ID, D_ID)
  storage::TableDef columns;

  struct tpcc_history tmp;

  columns.emplace_back("h_c_id", storage::INTEGER, sizeof(tmp.h_c_id));

  columns.emplace_back("h_c_d_id", storage::INTEGER, sizeof(tmp.h_c_d_id));
  columns.emplace_back("h_c_w_id", storage::INTEGER, sizeof(tmp.h_c_w_id));
  columns.emplace_back("h_d_id", storage::INTEGER, sizeof(tmp.h_d_id));
  columns.emplace_back("h_w_id", storage::INTEGER, sizeof(tmp.h_w_id));
  columns.emplace_back("h_date", storage::DATE, sizeof(tmp.h_date));
  columns.emplace_back("h_amount", storage::FLOAT, sizeof(tmp.h_amount));
  columns.emplace_back("h_data", storage::VARCHAR, sizeof(tmp.h_data));

  table_history = schema->create_table(
      "tpcc_history",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_history, false);
}

void TPCC::create_tbl_customer(uint64_t num_cust) {
  // Primary Key: (C_W_ID, C_D_ID, C_ID)
  // (C_W_ID, C_D_ID) Foreign Key, references (D_W_ID, D_ID)

  // TODO: get size of string vars from sizeof instead of hardcoded.

  struct tpcc_customer tmp;

  storage::TableDef columns;

  columns.emplace_back("c_id", storage::INTEGER, sizeof(tmp.c_id));

  columns.emplace_back("c_w_id", storage::INTEGER, sizeof(tmp.c_w_id));
  columns.emplace_back("c_d_id", storage::INTEGER, sizeof(tmp.c_d_id));

  columns.emplace_back("c_first", storage::VARCHAR, 17);
  columns.emplace_back("c_middle", storage::STRING, 2);
  columns.emplace_back("c_last", storage::VARCHAR, 17);

  columns.emplace_back("c_street_1", storage::VARCHAR, 21);
  columns.emplace_back("c_street_2", storage::VARCHAR, 21);
  columns.emplace_back("c_city", storage::VARCHAR, 21);
  columns.emplace_back("c_state", storage::STRING, 2);
  columns.emplace_back("c_zip", storage::STRING, 9);

  columns.emplace_back("c_phone", storage::STRING, 16);
  columns.emplace_back("c_since", storage::DATE, sizeof(tmp.c_since));

  columns.emplace_back("c_credit", storage::STRING, 2);
  columns.emplace_back("c_credit_lim", storage::FLOAT,
                       sizeof(tmp.c_credit_lim));
  columns.emplace_back("c_discount", storage::FLOAT, sizeof(tmp.c_discount));
  columns.emplace_back("c_balance", storage::FLOAT, sizeof(tmp.c_balance));
  columns.emplace_back("c_ytd_payment", storage::FLOAT,
                       sizeof(tmp.c_ytd_payment));

  columns.emplace_back("c_payment_cnt", storage::INTEGER,
                       sizeof(tmp.c_payment_cnt));
  columns.emplace_back("c_delivery_cnt", storage::INTEGER,
                       sizeof(tmp.c_delivery_cnt));
  columns.emplace_back("c_data", storage::VARCHAR, 501);
  columns.emplace_back("c_n_nationkey", storage::INTEGER,
                       sizeof(tmp.c_n_nationkey));
  table_customer = schema->create_table(
      "tpcc_customer",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_cust);
}

void TPCC::create_tbl_new_order(uint64_t num_new_order) {
  // Primary Key: (NO_W_ID, NO_D_ID, NO_O_ID)
  // (NO_W_ID, NO_D_ID, NO_O_ID) Foreign Key, references (O_W_ID, O_D_ID,
  // O_ID)

  struct tpcc_new_order tmp;

  storage::TableDef columns;

  columns.emplace_back("no_o_id", storage::INTEGER, sizeof(tmp.no_o_id));

  columns.emplace_back("no_d_id", storage::INTEGER, sizeof(tmp.no_d_id));
  columns.emplace_back("no_w_id", storage::INTEGER, sizeof(tmp.no_w_id));

  table_new_order = schema->create_table(
      "tpcc_neworder",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_new_order, index_on_order_tbl);
}

void TPCC::create_tbl_order(uint64_t num_order) {
  // Primary Key: (O_W_ID, O_D_ID, O_ID)
  // (O_W_ID, O_D_ID, O_C_ID) Foreign Key, references (C_W_ID, C_D_ID, C_ID)
  storage::TableDef columns;

  struct tpcc_order tmp;

  columns.emplace_back("o_id", storage::INTEGER, sizeof(tmp.o_id));

  columns.emplace_back("o_d_id", storage::INTEGER, sizeof(tmp.o_d_id));
  columns.emplace_back("o_w_id", storage::INTEGER, sizeof(tmp.o_w_id));
  columns.emplace_back("o_c_id", storage::INTEGER, sizeof(tmp.o_c_id));
  columns.emplace_back("o_entry_d", storage::DATE, sizeof(tmp.o_entry_d));
  columns.emplace_back("o_carrier_id", storage::INTEGER,
                       sizeof(tmp.o_carrier_id));
  columns.emplace_back("o_ol_cnt", storage::INTEGER, sizeof(tmp.o_ol_cnt));
  columns.emplace_back("o_all_local", storage::INTEGER,
                       sizeof(tmp.o_all_local));

  table_order = schema->create_table(
      "tpcc_order",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_order, index_on_order_tbl);
}

void TPCC::create_tbl_order_line(uint64_t num_order_line) {
  // Primary Key: (OL_W_ID, OL_D_ID, OL_O_ID, OL_NUMBER)
  // (OL_W_ID, OL_D_ID, OL_O_ID) Foreign Key, references (O_W_ID, O_D_ID,
  // O_ID)

  // (OL_SUPPLY_W_ID, OL_I_ID) Foreign Key, references (S_W_ID, S_I_ID)

  struct tpcc_orderline tmp = {};

  storage::TableDef columns;

  columns.emplace_back("ol_o_id", storage::INTEGER, sizeof(tmp.ol_o_id));

  columns.emplace_back("ol_d_id", storage::INTEGER, sizeof(tmp.ol_d_id));
  columns.emplace_back("ol_w_id", storage::INTEGER, sizeof(tmp.ol_w_id));
  columns.emplace_back("ol_number", storage::INTEGER, sizeof(tmp.ol_number));
  columns.emplace_back("ol_i_id", storage::INTEGER, sizeof(tmp.ol_i_id));
  columns.emplace_back("ol_supply_w_id", storage::INTEGER,
                       sizeof(tmp.ol_supply_w_id));
  columns.emplace_back("ol_delivery_d", storage::DATE,
                       sizeof(tmp.ol_delivery_d));
  columns.emplace_back("ol_quantity", storage::INTEGER,
                       sizeof(tmp.ol_quantity));
  columns.emplace_back("ol_amount", storage::FLOAT, sizeof(tmp.ol_amount));
  // columns.emplace_back(
  //     "ol_dist_info", storage::STRING, sizeof(tmp.ol_dist_info));

  table_order_line = schema->create_table(
      "tpcc_orderline",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_order_line, index_on_order_tbl);
}

void TPCC::create_tbl_supplier(uint64_t num_supp) {
  // Primary Key: suppkey
  /*
     uint32_t suppkey;
    char s_name[18];     // fix
    char s_address[41];  // var
    ushort s_nationkey;
    char s_phone[15];  // fix
    float s_acctbal;
    char s_comment[101];  // var
  */

  struct ch_supplier tmp;
  storage::TableDef columns;

  columns.emplace_back("su_suppkey", storage::INTEGER, sizeof(tmp.suppkey));

  columns.emplace_back("su_name", storage::STRING, sizeof(tmp.s_name));

  columns.emplace_back("su_address", storage::VARCHAR, sizeof(tmp.s_address));

  columns.emplace_back("su_nationkey", storage::INTEGER,
                       sizeof(tmp.s_nationkey));

  columns.emplace_back("su_phone", storage::STRING, sizeof(tmp.s_phone));

  columns.emplace_back("su_acctbal", storage::FLOAT, sizeof(tmp.s_acctbal));

  columns.emplace_back("su_comment", storage::VARCHAR, sizeof(tmp.s_comment));

  table_supplier = schema->create_table(
      "tpcc_supplier",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_supp, true, false);
}
void TPCC::create_tbl_region(uint64_t num_region) {
  // Primary Key: r_regionkey
  /*
     ushort r_regionkey;
      char r_name[12];      // var
      char r_comment[115];  // var
  */

  struct ch_region tmp;

  storage::TableDef columns;

  columns.emplace_back("r_regionkey", storage::INTEGER,
                       sizeof(tmp.r_regionkey));
  columns.emplace_back("r_name", storage::VARCHAR, sizeof(tmp.r_name));
  columns.emplace_back("r_comment", storage::VARCHAR, sizeof(tmp.r_comment));

  table_region = schema->create_table(
      "tpcc_region",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_region, true, false);
}
void TPCC::create_tbl_nation(uint64_t num_nation) {
  // Primary Key: n_nationkey
  /*
      ushort n_nationkey;
     char n_name[16];  // var
     ushort n_regionkey;
     char n_comment[115];  // var
  */
  struct ch_nation tmp;
  storage::TableDef columns;

  columns.emplace_back("n_nationkey", storage::INTEGER,
                       sizeof(tmp.n_nationkey));
  columns.emplace_back("n_name", storage::VARCHAR, sizeof(tmp.n_name));
  columns.emplace_back("n_regionkey", storage::INTEGER,
                       sizeof(tmp.n_regionkey));
  columns.emplace_back("n_comment", storage::VARCHAR, sizeof(tmp.n_comment));

  table_nation = schema->create_table(
      "tpcc_nation",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_nation, true, false);
}

/* A/C TPCC Specs*/
void TPCC::load_stock(int w_id, xid_t xid, partition_id_t partition_id,
                      master_version_t master_ver) {
  // Primary Key: (S_W_ID, S_I_ID)
  // S_W_ID Foreign Key, references W_ID
  // S_I_ID Foreign Key, references I_ID

  uint32_t base_sid = w_id * TPCC_MAX_ITEMS;

  struct tpcc_stock *stock_tmp = new struct tpcc_stock;
  //(struct tpcc_stock *)malloc(sizeof(struct tpcc_stock));

  int orig[TPCC_MAX_ITEMS], pos;
  for (int i = 0; i < TPCC_MAX_ITEMS / 10; i++) orig[i] = 0;
  for (int i = 0; i < TPCC_MAX_ITEMS / 10; i++) {
    do {
      pos = URand(&this->seed, 0L, TPCC_MAX_ITEMS - 1);
    } while (orig[pos]);
    orig[pos] = 1;
  }

  for (int i = 0; i < TPCC_MAX_ITEMS; i++) {
    uint32_t sid = base_sid + i;
    stock_tmp->s_i_id = i;
    stock_tmp->s_w_id = w_id;
    stock_tmp->s_quantity = URand(&this->seed, 10, 100);

    for (int j = 0; j < 10; j++) {
      make_alpha_string(&this->seed, 24, 24, stock_tmp->s_dist[j]);
    }

    stock_tmp->s_ytd = 0;
    stock_tmp->s_order_cnt = 0;
    stock_tmp->s_remote_cnt = 0;
    int data_len = make_alpha_string(&this->seed, 26, 50, stock_tmp->s_data);
    if (orig[i]) {
      int idx = URand(&this->seed, 0, data_len - 8);
      memcpy(&stock_tmp->s_data[idx], "original", 8);
    }

    // txn_id = 0, master_ver = 0
    void *hash_ptr =
        table_stock->insertRecord(stock_tmp, xid, partition_id, master_ver);
    this->table_stock->p_index->insert(sid, hash_ptr);
  }

  delete stock_tmp;
}

/* A/C TPCC Specs*/
void TPCC::load_item(int w_id, xid_t xid, partition_id_t partition_id,
                     master_version_t master_ver) {
  // Primary Key: I_ID

  struct tpcc_item item_temp;

  int orig[TPCC_MAX_ITEMS], pos;

  for (int i = 0; i < TPCC_MAX_ITEMS / 10; i++) orig[i] = 0;
  for (int i = 0; i < TPCC_MAX_ITEMS / 10; i++) {
    do {
      pos = URand(&this->seed, 0L, TPCC_MAX_ITEMS - 1);
    } while (orig[pos]);
    orig[pos] = 1;
  }

  for (uint32_t key = 0; key < TPCC_MAX_ITEMS; key++) {
    assert(key != TPCC_MAX_ITEMS);
    item_temp.i_id = key;
    item_temp.i_im_id = URand(&this->seed, 0L, TPCC_MAX_ITEMS - 1);

    make_alpha_string(&this->seed, 14, 24, item_temp.i_name);

    item_temp.i_price = ((double)URand(&this->seed, 100L, 10000L)) / 100.0;

    int data_len = make_alpha_string(&this->seed, 26, 50, item_temp.i_data);
    if (orig[key]) {
      int idx = URand(&this->seed, 0, data_len - 8);
      memcpy(&item_temp.i_data[idx], "original", 8);
    }

    void *hash_ptr = table_item->insertRecord(
        &item_temp, 0, key / (TPCC_MAX_ITEMS / g_num_partitions), 0);
    this->table_item->p_index->insert(key, hash_ptr);
  }
}

/* A/C TPCC Specs*/
void TPCC::load_district(int w_id, xid_t xid, partition_id_t partition_id,
                         master_version_t master_ver) {
  // Primary Key: (D_W_ID, D_ID) D_W_ID
  // Foreign Key, references W_ID

  struct tpcc_district *r = new struct tpcc_district;

  for (int d = 0; d < TPCC_NDIST_PER_WH; d++) {
    uint32_t dkey = MAKE_DIST_KEY(w_id, d);
    r->d_id = d;
    r->d_w_id = w_id;

    make_alpha_string(&this->seed, 6, 10, r->d_name);
    make_alpha_string(&this->seed, 10, 20, r->d_street[0]);
    make_alpha_string(&this->seed, 10, 20, r->d_street[1]);
    make_alpha_string(&this->seed, 10, 20, r->d_city);
    make_alpha_string(&this->seed, 2, 2, r->d_state);
    make_alpha_string(&this->seed, 9, 9, r->d_zip);
    r->d_tax = (double)URand(&this->seed, 10L, 20L) / 100.0;
    r->d_ytd = 30000.0;
    r->d_next_o_id = 3000;

    // std::cout << "%%%%%%%%%%" << std::endl;
    // char *dc = (char *)r;
    // dc += 97;
    // uint64_t *rr = (uint64_t *)(dc);
    // std::cout << "sending1: " << r->d_next_o_id << std::endl;
    // std::cout << "real offset: " << offsetof(struct tpcc_district,
    // d_next_o_id)
    //           << std::endl;
    // std::cout << "sending2: " << *rr << std::endl;
    // std::cout << "%%%%%%%%%%" << std::endl;

    void *hash_ptr =
        table_district->insertRecord(r, xid, partition_id, master_ver);
    this->table_district->p_index->insert((uint64_t)dkey, hash_ptr);
  }

  // std::cout << "offset1: " << offsetof(struct tpcc_district, d_id) <<
  // std::endl; std::cout << "offset2: " << offsetof(struct tpcc_district,
  // d_w_id)
  //           << std::endl;
  // std::cout << "offset3: " << offsetof(struct tpcc_district, d_name)
  //           << std::endl;
  // std::cout << "offset4: " << offsetof(struct tpcc_district, d_street)
  //           << std::endl;
  // std::cout << "offset5: " << offsetof(struct tpcc_district, d_street[1])
  //           << std::endl;
  // std::cout << "offset6: " << offsetof(struct tpcc_district, d_city)
  //           << std::endl;
  // std::cout << "offset7: " << offsetof(struct tpcc_district, d_state)
  //           << std::endl;
  // std::cout << "offset8: " << offsetof(struct tpcc_district, d_zip)
  //           << std::endl;
  // std::cout << "offset9: " << offsetof(struct tpcc_district, d_tax)
  //           << std::endl;
  // std::cout << "offset10: " << offsetof(struct tpcc_district, d_ytd)
  //           << std::endl;

  delete r;
}

/* A/C TPCC Specs*/
void TPCC::load_warehouse(int w_id, xid_t xid, partition_id_t partition_id,
                          master_version_t master_ver) {
  // Primary Key: W_ID
  struct tpcc_warehouse *w_temp = new struct tpcc_warehouse;

  w_temp->w_id = w_id;
  make_alpha_string(&this->seed, 6, 10, w_temp->w_name);
  make_alpha_string(&this->seed, 10, 20, w_temp->w_street[0]);
  make_alpha_string(&this->seed, 10, 20, w_temp->w_street[1]);
  make_alpha_string(&this->seed, 10, 20, w_temp->w_city);
  make_alpha_string(&this->seed, 2, 2, w_temp->w_state);
  make_alpha_string(&this->seed, 9, 9, w_temp->w_zip);
  w_temp->w_tax = (double)URand(&this->seed, 10L, 20L) / 100.0;
  w_temp->w_ytd = 300000.00;  // WRONG IN TPC-C SPECS!!

  // txn_id = 0, master_ver = 0
  void *hash_ptr =
      table_warehouse->insertRecord(w_temp, xid, partition_id, master_ver);
  assert(hash_ptr != nullptr);
  this->table_warehouse->p_index->insert(w_id, hash_ptr);
  delete w_temp;
}

/* A/C TPCC Specs*/
void TPCC::load_history(int w_id, xid_t xid, partition_id_t partition_id,
                        master_version_t master_ver) {
  // Primary Key: none
  // (H_C_W_ID, H_C_D_ID, H_C_ID) Foreign Key, references (C_W_ID, C_D_ID,
  // C_ID)
  // (H_W_ID, H_D_ID) Foreign Key, references (D_W_ID, D_ID)

  struct tpcc_history *r = new struct tpcc_history;

  for (int d = 0; d < TPCC_NDIST_PER_WH; d++) {
    for (int c = 0; c < TPCC_NCUST_PER_DIST; c++) {
      uint32_t key = MAKE_CUST_KEY(w_id, d, c);
      // key = MAKE_HASH_KEY(HISTORY_TID, pkey);

      // r = (struct tpcc_history *)e->value;
      r->h_c_id = c;
      r->h_c_d_id = d;
      r->h_c_w_id = w_id;
      r->h_d_id = d;
      r->h_w_id = w_id;
      r->h_date = get_timestamp();
      r->h_amount = 10.0;
      make_alpha_string(&this->seed, 12, 24, r->h_data);

      void *hash_ptr =
          table_history->insertRecord(r, xid, partition_id, master_ver);
    }
  }
  delete r;
}

void init_permutation(unsigned int *seed, uint64_t *cperm) {
  int i;

  for (i = 0; i < TPCC_NCUST_PER_DIST; i++) {
    cperm[i] = i + 1;
  }

  // shuffle
  for (i = 0; i < TPCC_NCUST_PER_DIST - 1; i++) {
    uint64_t j = URand(seed, i + 1, TPCC_NCUST_PER_DIST - 1);
    uint64_t tmp = cperm[i];
    cperm[i] = cperm[j];
    cperm[j] = tmp;
  }

  return;
}

/* A/C TPCC Specs*/
void TPCC::load_order(int w_id, xid_t xid, partition_id_t partition_id,
                      master_version_t master_ver) {
  // Order
  // Primary Key: (O_W_ID, O_D_ID, O_ID)
  // (O_W_ID, O_D_ID, O_C_ID) Foreign Key, references (C_W_ID, C_D_ID, C_ID)

  // Order-line
  // Primary Key: (OL_W_ID, OL_D_ID, OL_O_ID, OL_NUMBER)
  // (OL_W_ID, OL_D_ID, OL_O_ID) Foreign Key, references (O_W_ID, O_D_ID,
  // O_ID)

  // (OL_SUPPLY_W_ID, OL_I_ID) Foreign Key, references (S_W_ID, S_I_ID)

  uint64_t total_orderline_ins = 0;
  uint64_t total_order = 0;
  uint64_t order_per_wh = 0;
  uint64_t order_per_dist = 0;

  if (tpch_scale_factor != 0) {
    total_orderline_ins = 6001215 * tpch_scale_factor;
    total_order = total_orderline_ins / 15;
    order_per_wh = total_order / this->num_warehouse;
    order_per_dist = order_per_wh / TPCC_NDIST_PER_WH;
  } else {
    order_per_dist = TPCC_NCUST_PER_DIST;

    // total_order = TPCC_NCUST_PER_DIST * TPCC_NDIST_PER_WH;
    // order_per_wh = total_order / this->num_warehouses;
    // order_per_dist = order_per_wh / TPCC_NDIST_PER_WH;
    // total_orderline_ins = 6001215 * SF;
  }

  assert(order_per_dist < TPCC_MAX_ORD_PER_DIST);

  uint64_t pre_orders = (uint64_t)((double)order_per_dist * 0.7);

  uint64_t *cperm = (uint64_t *)malloc(sizeof(uint64_t) * TPCC_NCUST_PER_DIST);
  assert(cperm);

  std::vector<proteus::thread> loaders;

  for (int d = 0; d < TPCC_NDIST_PER_WH; d++) {
    init_permutation(&this->seed, cperm);

    //    loaders.emplace_back([this, d, order_per_dist, cperm, pre_orders,
    //    w_id, xid,
    //                          partition_id, master_ver]() {
    for (uint64_t o = 0; o < order_per_dist; o++) {
      struct tpcc_order r = {};  // new struct tpcc_order;

      uint64_t ckey = MAKE_ORDER_KEY(w_id, d, o);

      // if (ckey >= TPCC_MAX_ORDER_INITIAL_CAP) {
      //   std::cout << "w_id: " << w_id << std::endl;
      //   std::cout << " d_id: " << d << std::endl;
      //   std::cout << "o_id: " << o << std::endl;
      //   std::cout << "partition_id: " << partition_id << std::endl;
      //   std::cout << "ckey: " << ckey << std::endl;
      //   std::cout << "distk: " << MAKE_DIST_KEY(w_id, d) << std::endl;
      //   std::cout << "TPCC_MAX_ORD_PER_DIST: " << TPCC_MAX_ORD_PER_DIST
      //             << std::endl;
      //   std::cout << "-----------------" << std::endl;
      // }

      int c_id = cperm[o % TPCC_NCUST_PER_DIST];

      r.o_id = o;
      r.o_c_id = c_id;
      r.o_d_id = d;
      r.o_w_id = w_id;

      r.o_entry_d = get_timestamp();

      if (o < pre_orders) {
        // if (o < 2100) {
        r.o_carrier_id = URand(&this->seed, 1, 10);
      } else
        r.o_carrier_id = 0;

      int o_ol_cnt = TPCC_MAX_OL_PER_ORDER;  // URand(&this->seed, 5, 15);

      if (tpch_scale_factor != 0) {
        o_ol_cnt = TPCC_MAX_OL_PER_ORDER;
      }

      r.o_ol_cnt = o_ol_cnt;
      r.o_all_local = 1;

      // insert order here
      void *hash_ptr_o =
          table_order->insertRecord(&r, xid, partition_id, master_ver);
#if index_on_order_tbl
      assert(hash_ptr_o != nullptr);
      this->table_order->p_index->insert(ckey, hash_ptr_o);
#endif

      for (int ol = 0; ol < o_ol_cnt; ol++) {
        struct tpcc_orderline r_ol = {};  // new struct tpcc_orderline;

        uint64_t ol_pkey = MAKE_OL_KEY(w_id, d, o, ol);

        r_ol.ol_o_id = o;
        r_ol.ol_d_id = d;
        r_ol.ol_w_id = w_id;
        r_ol.ol_number = ol;
        r_ol.ol_i_id = URand(&this->seed, 0, TPCC_MAX_ITEMS - 1);
        r_ol.ol_supply_w_id = w_id;

        if (o < pre_orders) {
          r_ol.ol_delivery_d = r.o_entry_d;
          r_ol.ol_amount = 0;
        } else {
          r_ol.ol_delivery_d = 0;
          r_ol.ol_amount = ((double)URand(&this->seed, 1, 999999)) / 100.0;
        }
        r_ol.ol_quantity = 5;
        // make_alpha_string(&this->seed, 24, 24, r_ol.ol_dist_info);

        // insert orderline here
        void *hash_ptr_ol = table_order_line->insertRecord(
            &r_ol, xid, partition_id, master_ver);
#if index_on_order_tbl
        assert(hash_ptr_ol != nullptr);
        //          LOG(INFO) << "X: " << ol_pkey << " | w_id: " << (uint)w_id
        //          <<" |d: " << (uint)d << " | o: " << (uint)o << " |ol: " <<
        //          (uint)ol;
        this->table_order_line->p_index->insert(ol_pkey, hash_ptr_ol);
#endif
      }

      // NEW ORDER
      if (o >= pre_orders) {
        struct tpcc_new_order r_no = {};  // new struct tpcc_new_order;

        r_no.no_o_id = o;
        r_no.no_d_id = d;
        r_no.no_w_id = w_id;
        // insert new order here

        void *hash_ptr_no =
            table_new_order->insertRecord(&r_no, xid, partition_id, master_ver);
#if index_on_order_tbl
        assert(hash_ptr_no != nullptr || hash_ptr_no != NULL);
        this->table_new_order->p_index->insert(ckey, hash_ptr_no);
#endif
      }
    }
    //});
  }

  for (auto &th : loaders) {
    th.join();
  }

  // delete r;
  // delete r_ol;
  // delete r_no;
  free(cperm);
}

int TPCC::set_last_name(int num, char *name) {
  static const char *n[] = {"BAR", "OUGHT", "ABLE",  "PRI",   "PRES",
                            "ESE", "ANTI",  "CALLY", "ATION", "EING"};

  strcpy(name, n[num / 100]);
  strcat(name, n[(num / 10) % 10]);
  strcat(name, n[num % 10]);
  return strlen(name);
}

uint64_t TPCC::cust_derive_key(const char *c_last, int c_d_id, int c_w_id) {
  uint64_t key = 0;
  char offset = 'A';
  for (uint32_t i = 0; i < strlen(c_last); i++)
    key = (key << 1) + (c_last[i] - offset);
  key = key << 10;
  key += c_w_id * TPCC_NDIST_PER_WH + c_d_id;

  return key;
}

/* A/C TPCC Specs*/
void TPCC::load_customer(int w_id, xid_t xid, partition_id_t partition_id,
                         master_version_t master_ver) {
  // Primary Key: (C_W_ID, C_D_ID, C_ID)
  // (C_W_ID, C_D_ID) Foreign Key, references (D_W_ID, D_ID)

  // void *hash_ptr = table_customer->insertRecord(r, 0, 0);
  // this->table_customer->p_index->insert(key, hash_ptr);

  struct tpcc_customer *r = new tpcc_customer;
  for (uint64_t d = 0; d < TPCC_NDIST_PER_WH; d++) {
    for (uint64_t c = 0; c < TPCC_NCUST_PER_DIST; c++) {
      uint64_t ckey = MAKE_CUST_KEY(w_id, d, c);

      // r = (struct tpcc_customer *)e->value;
      r->c_id = c;
      r->c_d_id = d;
      r->c_w_id = w_id;

      if (c < 1000)
        set_last_name(c, r->c_last);
      else
        set_last_name(NURand(&this->seed, 255, 0, 999), r->c_last);

      memcpy(r->c_middle, "OE", 2);

      make_alpha_string(&this->seed, FIRST_NAME_MIN_LEN, FIRST_NAME_LEN,
                        r->c_first);

      make_alpha_string(&this->seed, 10, 20, r->c_street[0]);
      make_alpha_string(&this->seed, 10, 20, r->c_street[1]);
      make_alpha_string(&this->seed, 10, 20, r->c_city);
      make_alpha_string(&this->seed, 2, 2, r->c_state);     /* State */
      make_numeric_string(&this->seed, 9, 9, r->c_zip);     /* Zip */
      make_numeric_string(&this->seed, 16, 16, r->c_phone); /* Zip */
      r->c_since = get_timestamp();
      r->c_credit_lim = 50000;
      r->c_delivery_cnt = 0;
      make_alpha_string(&this->seed, 300, 500, r->c_data);

      if (RAND(&this->seed, 10) == 0) {
        r->c_credit[0] = 'G';
      } else {
        r->c_credit[0] = 'B';
      }
      r->c_credit[1] = 'C';
      r->c_discount = (double)RAND(&this->seed, 5000) / 10000;
      r->c_balance = -10.0;
      r->c_ytd_payment = 10.0;
      r->c_payment_cnt = 1;

      void *hash_ptr =
          table_customer->insertRecord(r, xid, partition_id, master_ver);
      this->table_customer->p_index->insert(ckey, hash_ptr);

      /* create secondary index using the main hash table itself.
       * we can do this by deriving a key from the last name,dist,wh id
       * and using it to create a new record which will contain both
       * the real key of all records with that last name
       * XXX: Note that this key is not unique - so all names hashing to
       * the same key will hash to the same key. Thus, ppl with different
       * last names might hash to the same sr record.
       */
      uint64_t sr_dkey = cust_derive_key(r->c_last, d, w_id);

      // pull up the record if its already there
      struct secondary_record sr;
      int sr_idx, sr_nids;

      if (cust_sec_index->find(sr_dkey, sr)) {
        sr_idx = sr.sr_idx;
        sr_nids = sr.sr_nids;
      } else {
        // sie = hash_insert(p, sr_key, sizeof(struct secondary_record),
        // NULL); assert(sie);

        // sr = (struct secondary_record *)sie->value;

        /* XXX: memory leak possibility - if this record is ever freed
         * this malloc wont be released
         */
        sr.sr_rids = (uint32_t *)malloc(sizeof(uint64_t) * NDEFAULT_RIDS);
        assert(sr.sr_rids);
        sr.sr_idx = sr_idx = 0;
        sr.sr_nids = sr_nids = NDEFAULT_RIDS;
        cust_sec_index->insert(sr_dkey, sr);
      }

      assert(sr_idx < sr_nids);

      /* add this record to the index */
      sr.sr_rids[sr_idx] = sr_dkey;
      if (++sr_idx == sr_nids) {
        // reallocate the record array
        sr_nids *= 2;
        sr.sr_rids =
            (uint32_t *)realloc(sr.sr_rids, sizeof(uint64_t) * sr_nids);
        assert(sr.sr_rids);
      }

      sr.sr_idx = sr_idx;
      sr.sr_nids = sr_nids;

      cust_sec_index->update(sr_dkey, sr);
    }
  }
  delete r;
}

void TPCC::load_nation(int w_id, xid_t xid, partition_id_t partition_id,
                       master_version_t master_ver) {
  struct Nation {
    int id;
    std::string name;
    int rId;
  };

  const Nation nations[] = {{48, "ALGERIA", 0},       {49, "ARGENTINA", 1},
                            {50, "BRAZIL", 1},        {51, "CANADA", 1},
                            {52, "EGYPT", 4},         {53, "ETHIOPIA", 0},
                            {54, "FRANCE", 3},        {55, "GERMANY", 3},
                            {56, "INDIA", 2},         {57, "INDONESIA", 2},
                            {65, "IRAN", 4},          {66, "IRAQ", 4},
                            {67, "JAPAN", 2},         {68, "JORDAN", 4},
                            {69, "KENYA", 0},         {70, "MOROCCO", 0},
                            {71, "MOZAMBIQUE", 0},    {72, "PERU", 1},
                            {73, "CHINA", 2},         {74, "ROMANIA", 3},
                            {75, "SAUDI ARABIA", 4},  {76, "VIETNAM", 2},
                            {77, "RUSSIA", 3},        {78, "UNITED KINGDOM", 3},
                            {79, "UNITED STATES", 1}, {80, "CHINA", 2},
                            {81, "PAKISTAN", 2},      {82, "BANGLADESH", 2},
                            {83, "MEXICO", 1},        {84, "PHILIPPINES", 2},
                            {85, "THAILAND", 2},      {86, "ITALY", 3},
                            {87, "SOUTH AFRICA", 0},  {88, "SOUTH KOREA", 2},
                            {89, "COLOMBIA", 1},      {90, "SPAIN", 3},
                            {97, "UKRAINE", 3},       {98, "POLAND", 3},
                            {99, "SUDAN", 0},         {100, "UZBEKISTAN", 2},
                            {101, "MALAYSIA", 2},     {102, "VENEZUELA", 1},
                            {103, "NEPAL", 2},        {104, "AFGHANISTAN", 2},
                            {105, "NORTH KOREA", 2},  {106, "TAIWAN", 2},
                            {107, "GHANA", 0},        {108, "IVORY COAST", 0},
                            {109, "SYRIA", 4},        {110, "MADAGASCAR", 0},
                            {111, "CAMEROON", 0},     {112, "SRI LANKA", 2},
                            {113, "ROMANIA", 3},      {114, "NETHERLANDS", 3},
                            {115, "CAMBODIA", 2},     {116, "BELGIUM", 3},
                            {117, "GREECE", 3},       {118, "PORTUGAL", 3},
                            {119, "ISRAEL", 4},       {120, "FINLAND", 3},
                            {121, "SINGAPORE", 2},    {122, "NORWAY", 3}};

  // Nation
  for (int i = 0; i < 62; i++) {
    struct ch_nation ins = {};

    memcpy(ins.n_name, nations[i].name.c_str(), 16);
    ins.n_nationkey = nations[i].id;
    ins.n_regionkey = nations[i].rId;

    // TODO: from ch-benchmark.
    // ins.n_comment = ;

    void *hash_ptr =
        table_nation->insertRecord(&ins, xid, partition_id, master_ver);
    this->table_nation->p_index->insert(ins.n_nationkey, hash_ptr);
  }
}

void TPCC::load_region(int w_id, xid_t xid, partition_id_t partition_id,
                       master_version_t master_ver) {
  const char *regions[] = {"AFRICA", "AMERICA", "ASIA", "EUROPE",
                           "MIDDLE EAST"};
  // Region
  for (int rId = 0; rId < 5; rId++) {
    struct ch_region ins = {};

    memcpy(ins.r_name, regions[rId], 12);
    ins.r_regionkey = rId;

    // TODO: from ch-benchmark.
    // ins.r_comment;

    void *hash_ptr =
        table_region->insertRecord(&ins, xid, partition_id, master_ver);
    this->table_region->p_index->insert(rId, hash_ptr);
  }
}

void TPCC::load_supplier(int w_id, xid_t xid, partition_id_t partition_id,
                         master_version_t master_ver) {
  // Supplier
  for (int suId = 0; suId < 10000; suId++) {
    struct ch_supplier supp_ins = {};

    supp_ins.suppkey = suId;

    stringstream ss;
    ss << "Supplier#" << std::setw(9) << std::setfill('0') << suId;

    strcpy(supp_ins.s_name, ss.str().c_str());
    make_alpha_string(&this->seed, 10, 40, supp_ins.s_address);

    int rand = 0;
    while (rand == 0 || (rand > '9' && rand < 'A') ||
           (rand > 'Z' && rand < 'a')) {
      rand = URand(&this->seed, '0', 'z');
    }
    supp_ins.s_nationkey = rand;

    stringstream suPhn;

    int country_code = (suId % 90) + 10;  // ensure length 2
    suPhn << country_code << "-";
    suPhn << URand(&this->seed, 100, 999);
    suPhn << "-";
    suPhn << URand(&this->seed, 100, 999);
    suPhn << "-";
    suPhn << URand(&this->seed, 100, 999);

    strcpy(supp_ins.s_phone, suPhn.str().c_str());
    make_alpha_string(&this->seed, 10, 40, supp_ins.s_address);
    supp_ins.s_acctbal = (double)URand(&this->seed, -99999, 999999) / 100.0;

    // TODO: from ch-benchmark.
    // char s_comment[101];

    void *hash_ptr =
        table_supplier->insertRecord(&supp_ins, xid, partition_id, master_ver);
    this->table_supplier->p_index->insert(suId, hash_ptr);
  }
}

void TPCC::pre_run(worker_id_t wid, xid_t xid, partition_id_t partition_id,
                   master_version_t master_ver) {
  // static std::mutex print_mutex;
  // {
  //   std::unique_lock<std::mutex> lk(print_mutex);
  //   std::cout << "pre-run-------------------------------" << std::endl;
  //   std::cout << "pid: " << partition_id << std::endl;
  //   std::cout << "wid: " << wid << std::endl;
  // }
  if (wid >= this->num_warehouse) return;

  assert(partition_id < g_num_partitions);

  if (wid == 0) load_item(wid, xid, partition_id, master_ver);

  if (wid == 0 && this->is_ch_benchmark) {
    load_region(wid, xid, partition_id, master_ver);
    load_supplier(wid, xid, partition_id, master_ver);
    load_nation(wid, xid, partition_id, master_ver);
  }

  load_warehouse(wid, xid, partition_id, master_ver);

  load_district(wid, xid, partition_id, master_ver);

  load_stock(wid, xid, partition_id, master_ver);

  load_history(wid, xid, partition_id, master_ver);

  load_customer(wid, xid, partition_id, master_ver);

#if !debug_dont_load_order
  load_order(wid, xid, partition_id, master_ver);
#endif
}

std::ostream &operator<<(std::ostream &out, const TPCC::ch_nation &r) {
  out << r.n_nationkey << csv_delim;
  out << r.n_name << csv_delim;
  out << r.n_regionkey << csv_delim;
  out << r.n_comment << csv_delim;
  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::ch_region &r) {
  out << r.r_regionkey << csv_delim;
  out << r.r_name << csv_delim;
  out << r.r_comment << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::ch_supplier &r) {
  out << r.suppkey << csv_delim;
  out << r.s_name << csv_delim;
  out << r.s_address << csv_delim;
  out << r.s_nationkey << csv_delim;
  out << r.s_phone << csv_delim;
  out << r.s_acctbal << csv_delim;
  out << r.s_comment << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_stock &r) {
  out << r.s_i_id << csv_delim;
  out << r.s_w_id << csv_delim;
  out << r.s_quantity << csv_delim;
  for (size_t i = 0; i < TPCC_NDIST_PER_WH; i++) {
    out << r.s_dist[i] << csv_delim;
  }
  out << r.s_ytd << csv_delim;
  out << r.s_order_cnt << csv_delim;
  out << r.s_remote_cnt << csv_delim;
  out << r.s_data << csv_delim;
  out << r.s_su_suppkey << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_item &r) {
  out << r.i_id << csv_delim;
  out << r.i_im_id << csv_delim;
  out << r.i_name << csv_delim;
  out << r.i_price << csv_delim;
  out << r.i_data << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_warehouse &r) {
  out << r.w_id << csv_delim;
  out << r.w_name << csv_delim;
  out << r.w_street[0] << csv_delim;
  out << r.w_street[1] << csv_delim;
  out << r.w_city << csv_delim;
  out << r.w_state << csv_delim;
  out << r.w_zip << csv_delim;
  out << r.w_tax << csv_delim;
  out << r.w_ytd << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_district &r) {
  out << r.d_id << csv_delim;
  out << r.d_w_id << csv_delim;
  out << r.d_name << csv_delim;
  out << r.d_street[0] << csv_delim;
  out << r.d_street[1] << csv_delim;
  out << r.d_city << csv_delim;
  out << r.d_state << csv_delim;
  out << r.d_zip << csv_delim;
  out << r.d_tax << csv_delim;
  out << r.d_ytd << csv_delim;
  out << r.d_next_o_id << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_history &r) {
  out << r.h_c_id << csv_delim;
  out << r.h_c_d_id << csv_delim;
  out << r.h_c_w_id << csv_delim;
  out << r.h_d_id << csv_delim;
  out << r.h_w_id << csv_delim;
  out << r.h_date << csv_delim;
  out << r.h_amount << csv_delim;
  out << r.h_data << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_customer &r) {
  out << r.c_id << csv_delim;
  out << r.c_w_id << csv_delim;
  out << r.c_d_id << csv_delim;
  out << r.c_first << csv_delim;
  out << r.c_middle << csv_delim;
  out << r.c_last << csv_delim;
  out << r.c_street[0] << csv_delim;
  out << r.c_street[1] << csv_delim;
  out << r.c_city << csv_delim;
  out << r.c_state << csv_delim;
  out << r.c_zip << csv_delim;
  out << r.c_phone << csv_delim;
  out << r.c_since << csv_delim;
  out << r.c_credit << csv_delim;
  out << r.c_credit_lim << csv_delim;
  out << r.c_discount << csv_delim;
  out << r.c_balance << csv_delim;
  out << r.c_ytd_payment << csv_delim;
  out << r.c_payment_cnt << csv_delim;
  out << r.c_delivery_cnt << csv_delim;
  out << r.c_data << csv_delim;
  out << r.c_n_nationkey << csv_delim;
  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_order &r) {
  out << r.o_id << csv_delim;
  out << r.o_d_id << csv_delim;
  out << r.o_w_id << csv_delim;
  out << r.o_c_id << csv_delim;
  out << r.o_entry_d << csv_delim;
  out << r.o_carrier_id << csv_delim;
  out << r.o_ol_cnt << csv_delim;
  out << r.o_all_local << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_orderline &r) {
  out << r.ol_o_id << csv_delim;
  out << r.ol_d_id << csv_delim;
  out << r.ol_w_id << csv_delim;
  out << r.ol_number << csv_delim;
  out << r.ol_i_id << csv_delim;
  out << r.ol_supply_w_id << csv_delim;
  out << r.ol_delivery_d << csv_delim;
  out << r.ol_quantity << csv_delim;
  out << r.ol_amount << csv_delim;
  // out << r.ol_dist_info << csv_delim;

  out << std::endl;
  return out;
}
std::ostream &operator<<(std::ostream &out, const TPCC::tpcc_new_order &r) {
  out << r.no_o_id << csv_delim;
  out << r.no_d_id << csv_delim;
  out << r.no_w_id << csv_delim;

  out << std::endl;
  return out;
}
}  // namespace bench
