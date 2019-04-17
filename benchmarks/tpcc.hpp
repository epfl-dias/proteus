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
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
//#include <thread

namespace bench {

/*

        Benchmark: TPC-C

        Description:

        TODO: PK of tables..

*/

class TPCC : public Benchmark {
 private:
  storage::Schema *schema;
  storage::Table *warehouse;
  storage::Table *district;
  storage::Table *customer;
  storage::Table *history;
  storage::Table *new_order;
  storage::Table *order;
  storage::Table *order_line;
  storage::Table *item;
  storage::Table *stock;

  // std::atomic<bool> initialized;  // so that nobody calls load twice
  // std::atomic<uint64_t> key_gen;

 public:
  void create_tbl_warehouse(int num_warehouses = 1) {
    // Primary Key: W_ID
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_id", storage::INTEGER, sizeof(short)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_name", storage::VARCHAR, 11));  // size 10 +1 for null character

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_street_1", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_street_2", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_city", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_state", storage::STRING, 2));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_zip", storage::STRING, 9));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_tax", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_ytd", storage::FLOAT, sizeof(float)));

    warehouse = schema->create_table("tpcc_warehouse", storage::COLUMN_STORE,
                                     columns, num_warehouses);
  }

  void create_tbl_district(int num_districts = 1) {
    // Primary Key: (D_W_ID, D_ID) D_W_ID
    // Foreign Key, references W_ID
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_name", storage::VARCHAR, 11));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_street_1", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_street_2", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_city", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_state", storage::STRING, 2));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_zip", storage::STRING, 9));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_tax", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "w_ytd", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "d_next_o_id", storage::INTEGER, sizeof(uint64_t)));

    district = schema->create_table("tpcc_district", storage::COLUMN_STORE,
                                    columns, num_districts);
  }

  void create_tbl_customer(int num_cust = 1) {
    // Primary Key: (C_W_ID, C_D_ID, C_ID)
    // (C_W_ID, C_D_ID) Foreign Key, references (D_W_ID, D_ID)
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_id", storage::INTEGER, sizeof(uint32_t)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_d_id", storage::INTEGER, sizeof(ushort)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_first", storage::VARCHAR, 17));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_middle", storage::STRING, 2));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_last", storage::VARCHAR, 17));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_street_1", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_street_2", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_city", storage::VARCHAR, 21));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_state", storage::STRING, 2));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_zip", storage::STRING, 9));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_phone", storage::STRING, 16));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_since", storage::DATE, sizeof(uint32_t)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_credit", storage::STRING, 2));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_credit_lim", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_discount", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_balance", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_ytd_payment", storage::FLOAT, sizeof(float)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_payment_cnt", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_delivery_cnt", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "c_data", storage::VARCHAR, 501));

    customer = schema->create_table("tpcc_customer", storage::COLUMN_STORE,
                                    columns, num_cust);
  }

  void create_tbl_history(int num_history = 1) {
    // Primary Key: none
    // (H_C_W_ID, H_C_D_ID, H_C_ID) Foreign Key, references (C_W_ID, C_D_ID,
    // C_ID)
    // (H_W_ID, H_D_ID) Foreign Key, references (D_W_ID, D_ID)
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_c_id", storage::INTEGER, sizeof(uint32_t)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_c_d_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_c_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_d_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_date", storage::DATE, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_amount", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "h_data", storage::VARCHAR, 25));

    history = schema->create_table("tpcc_history", storage::COLUMN_STORE,
                                   columns, num_history);
  }

  void create_tbl_new_order(int num_new_order = 1) {
    // Primary Key: (NO_W_ID, NO_D_ID, NO_O_ID)
    // (NO_W_ID, NO_D_ID, NO_O_ID) Foreign Key, references (O_W_ID, O_D_ID,
    // O_ID)
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "no_o_id", storage::INTEGER, sizeof(uint64_t)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "no_d_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "no_w_id", storage::INTEGER, sizeof(ushort)));

    new_order = schema->create_table("tpcc_new_order", storage::COLUMN_STORE,
                                     columns, num_new_order);
  }

  void create_tbl_order(int num_order = 1) {
    // Primary Key: (O_W_ID, O_D_ID, O_ID)
    // (O_W_ID, O_D_ID, O_C_ID) Foreign Key, references (C_W_ID, C_D_ID, C_ID)
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_id", storage::INTEGER, sizeof(uint64_t)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_d_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_c_id", storage::INTEGER, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_entry_date", storage::DATE, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_carrier_id", storage::INTEGER, sizeof(short)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_ol_cnt", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "o_all_local", storage::INTEGER, sizeof(ushort)));
    order = schema->create_table("tpcc_order", storage::COLUMN_STORE, columns,
                                 num_order);
  }

  void create_tbl_order_line(int num_order_line = 1) {
    // Primary Key: (OL_W_ID, OL_D_ID, OL_O_ID, OL_NUMBER)
    // (OL_W_ID, OL_D_ID, OL_O_ID) Foreign Key, references (O_W_ID, O_D_ID,
    // O_ID)

    // (OL_SUPPLY_W_ID, OL_I_ID) Foreign Key, references (S_W_ID, S_I_ID)
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_o_id", storage::INTEGER, sizeof(uint64_t)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_d_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_number", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_i_id", storage::INTEGER, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_supply_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_delivery_d", storage::DATE, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_quantity", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_amount", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "ol_dist_info", storage::STRING, 24));

    order_line = schema->create_table("tpcc_order_line", storage::COLUMN_STORE,
                                      columns, num_order_line);
  }

  void create_tbl_item(int num_item = 1) {
    // Primary Key: I_ID
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "i_id", storage::INTEGER, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "i_im_id", storage::INTEGER, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "i_name", storage::VARCHAR, 25));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "i_price", storage::FLOAT, sizeof(float)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "i_data", storage::VARCHAR, 51));

    item = schema->create_table("tpcc_item", storage::COLUMN_STORE, columns,
                                num_item);
  }

  void create_tbl_stock(int num_stock = 1) {
    // Primary Key: (S_W_ID, S_I_ID)
    // S_W_ID Foreign Key, references W_ID
    // S_I_ID Foreign Key, references I_ID
    std::vector<std::tuple<std::string, storage::data_type, size_t> > columns;

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_i_id", storage::INTEGER, sizeof(uint32_t)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_w_id", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_quantity", storage::FLOAT, sizeof(short)));

    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_01", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_02", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_03", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_04", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_05", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_06", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_07", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_08", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_09", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_dist_10", storage::STRING, 24));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_ytd", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_order_cnt", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_remote_cnt", storage::INTEGER, sizeof(ushort)));
    columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
        "s_data", storage::VARCHAR, 51));

    stock = schema->create_table("tpcc_stock", storage::COLUMN_STORE, columns,
                                 num_stock);
  }

  void load_data(int num_threads = 1){};

  // struct txn::TXN gen_insert_txn(uint64_t key, void *rec) {}
  // struct txn::TXN gen_upd_txn(uint64_t key, void *rec) {}

  // void *gen_txn(int wid) {}

  // void exec_txn(void *stmts) { return; }

  // TODO: clean-up
  ~TPCC() {}

  // private:
  TPCC(std::string name = "TPCC") : Benchmark(name) {
    this->schema = &storage::Schema::getInstance();
  };
};

}  // namespace bench

#endif /* TPCC_HPP_ */
