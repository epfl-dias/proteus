#include "benchmarks/tpcc.hpp"

namespace bench {

TPCC::TPCC(std::string name, int num_warehouses)
    : Benchmark(name), num_warehouse(num_warehouses) {
  this->schema = &storage::Schema::getInstance();
  this->seed = rand();

  uint64_t total_districts = TPCC_NDIST_PER_WH * this->num_warehouse;
  uint64_t max_customers = TPCC_NCUST_PER_DIST * total_districts;
  uint64_t max_orders = 10000000;
  uint64_t max_order_line = TPCC_MAX_OL_PER_ORDER * max_orders;
  uint64_t max_stock = TPCC_MAX_ITEMS * this->num_warehouse;

  this->create_tbl_warehouse(this->num_warehouse);
  this->create_tbl_district(total_districts);
  this->create_tbl_customer(max_customers);
  this->create_tbl_history(max_customers);
  this->create_tbl_new_order(max_orders);
  this->create_tbl_order(max_orders);
  this->create_tbl_order_line(max_order_line);
  this->create_tbl_item(TPCC_MAX_ITEMS);
  this->create_tbl_stock(max_stock);

  cust_sec_index = new indexes::HashIndex<uint64_t, struct secondary_record>();
  cust_sec_index->reserve(max_customers);
}

void TPCC::create_tbl_warehouse(uint64_t num_warehouses) {
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

  table_warehouse = schema->create_table(
      "tpcc_warehouse", storage::COLUMN_STORE, columns, num_warehouses);
}

void TPCC::create_tbl_district(uint64_t num_districts) {
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

  table_district = schema->create_table("tpcc_district", storage::COLUMN_STORE,
                                        columns, num_districts);
}

void TPCC::create_tbl_item(uint64_t num_item) {
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

  table_item = schema->create_table("tpcc_item", storage::COLUMN_STORE, columns,
                                    num_item);
}

void TPCC::create_tbl_stock(uint64_t num_stock) {
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

  table_stock = schema->create_table("tpcc_stock", storage::COLUMN_STORE,
                                     columns, num_stock);
}

void TPCC::create_tbl_history(uint64_t num_history) {
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

  table_history = schema->create_table("tpcc_history", storage::COLUMN_STORE,
                                       columns, num_history);
}

void TPCC::create_tbl_customer(uint64_t num_cust) {
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

  table_customer = schema->create_table("tpcc_customer", storage::COLUMN_STORE,
                                        columns, num_cust);
}

void TPCC::create_tbl_new_order(uint64_t num_new_order) {
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

  table_new_order = schema->create_table(
      "tpcc_new_order", storage::COLUMN_STORE, columns, num_new_order);
}

void TPCC::create_tbl_order(uint64_t num_order) {
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
  table_order = schema->create_table("tpcc_order", storage::COLUMN_STORE,
                                     columns, num_order);
}

void TPCC::create_tbl_order_line(uint64_t num_order_line) {
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

  table_order_line = schema->create_table(
      "tpcc_order_line", storage::COLUMN_STORE, columns, num_order_line);
}

void TPCC::load_stock(int w_id) {
  // Primary Key: (S_W_ID, S_I_ID)
  // S_W_ID Foreign Key, references W_ID
  // S_I_ID Foreign Key, references I_ID

  uint32_t base_sid = w_id * TPCC_MAX_ITEMS;

  struct tpcc_stock *stock_tmp = new struct tpcc_stock;
  //(struct tpcc_stock *)malloc(sizeof(struct tpcc_stock));

  // WHY START FROM 1 BELOW ??
  for (int i = 1; i <= TPCC_MAX_ITEMS; i++) {
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
    int len = make_alpha_string(&this->seed, 26, 50, stock_tmp->s_data);
    if (RAND(&this->seed, 100) < 10) {
      int idx = URand(&this->seed, 0, len - 8);
      memcpy(&stock_tmp->s_data[idx], "original", 8);
    }
    // txn_id = 0, master_ver = 0
    void *hash_ptr = table_stock->insertRecord(stock_tmp, 0, 0);
    this->table_stock->p_index->insert(sid, hash_ptr);
  }

  delete stock_tmp;
}

void TPCC::load_item(int w_id) {
  // Primary Key: I_ID

  struct tpcc_item *item_temp = new struct tpcc_item;
  assert(w_id == 1);

  for (uint32_t key = 1; key <= TPCC_MAX_ITEMS; key++) {
    item_temp->i_id = key;
    item_temp->i_im_id = URand(&this->seed, 1L, 10000L);
    make_alpha_string(&this->seed, 14, 24, item_temp->i_name);
    item_temp->i_price = URand(&this->seed, 1, 100);
    int data_len = make_alpha_string(&this->seed, 26, 50, item_temp->i_data);

    // TODO in TPCC, "original" should start at a random position
    if (RAND(&this->seed, 10) == 0) {
      int idx = URand(&this->seed, 0, data_len - 8);
      memcpy(&item_temp->i_data[idx], "original", 8);
    }

    void *hash_ptr = table_item->insertRecord(item_temp, 0,
                                              0);  // txn_id = 0, master_ver = 0
    this->table_item->p_index->insert(key, hash_ptr);
  }
  delete item_temp;
}

void TPCC::load_district(int w_id) {
  // Primary Key: (D_W_ID, D_ID) D_W_ID
  // Foreign Key, references W_ID

  struct tpcc_district *r = new struct tpcc_district;

  for (int d = 1; d <= TPCC_NDIST_PER_WH; d++) {
    ushort dkey = MAKE_DIST_KEY(w_id, d);
    r->d_id = d;
    r->d_w_id = w_id;

    make_alpha_string(&this->seed, 6, 10, r->d_name);
    make_alpha_string(&this->seed, 10, 20, r->d_street[0]);
    make_alpha_string(&this->seed, 10, 20, r->d_street[1]);
    make_alpha_string(&this->seed, 10, 20, r->d_city);
    make_alpha_string(&this->seed, 2, 2, r->d_state);
    make_alpha_string(&this->seed, 9, 9, r->d_zip);
    float tax = (float)URand(&this->seed, 0L, 200L) / 1000.0;
    float w_ytd = 30000.00;
    r->d_tax = tax;
    r->d_ytd = w_ytd;
    r->d_next_o_id = 3001;

    void *hash_ptr = table_district->insertRecord(r, 0, 0);
    this->table_district->p_index->insert(dkey, hash_ptr);
  }

  delete r;
}

void TPCC::load_warehouse(int w_id) {
  // Primary Key: W_ID
  struct tpcc_warehouse *w_temp = new struct tpcc_warehouse;

  w_temp->w_id = w_id;
  make_alpha_string(&this->seed, 6, 10, w_temp->w_name);
  make_alpha_string(&this->seed, 10, 20, w_temp->w_street[0]);
  make_alpha_string(&this->seed, 10, 20, w_temp->w_street[1]);
  make_alpha_string(&this->seed, 10, 20, w_temp->w_city);
  make_alpha_string(&this->seed, 2, 2, w_temp->w_state);
  make_alpha_string(&this->seed, 9, 9, w_temp->w_zip);
  float tax = (float)URand(&this->seed, 0L, 200L) / 1000.0;
  float w_ytd = 300000.00;
  w_temp->w_tax = tax;
  w_temp->w_ytd = w_ytd;

  // txn_id = 0, master_ver = 0
  void *hash_ptr = table_warehouse->insertRecord(w_temp, 0, 0);
  this->table_warehouse->p_index->insert(w_id, hash_ptr);
  delete w_temp;
}

void TPCC::load_history(int w_id) {
  // Primary Key: none
  // (H_C_W_ID, H_C_D_ID, H_C_ID) Foreign Key, references (C_W_ID, C_D_ID,
  // C_ID)
  // (H_W_ID, H_D_ID) Foreign Key, references (D_W_ID, D_ID)

  struct tpcc_history *r = new struct tpcc_history;

  for (int d = 1; d <= TPCC_NDIST_PER_WH; d++) {
    for (int c = 1; c <= TPCC_NCUST_PER_DIST; c++) {
      uint32_t key = MAKE_CUST_KEY(w_id, d, c);
      // key = MAKE_HASH_KEY(HISTORY_TID, pkey);

      // r = (struct tpcc_history *)e->value;
      r->h_c_id = c;
      r->h_c_d_id = d;
      r->h_c_w_id = w_id;
      r->h_d_id = d;
      r->h_w_id = w_id;
      r->h_date = 0;
      r->h_amount = 10.0;
      make_alpha_string(&this->seed, 12, 24, r->h_data);

      void *hash_ptr = table_history->insertRecord(r, 0, 0);
      this->table_history->p_index->insert(key, hash_ptr);
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

void TPCC::load_order(int w_id) {
  // Order
  // Primary Key: (O_W_ID, O_D_ID, O_ID)
  // (O_W_ID, O_D_ID, O_C_ID) Foreign Key, references (C_W_ID, C_D_ID, C_ID)

  // Order-line
  // Primary Key: (OL_W_ID, OL_D_ID, OL_O_ID, OL_NUMBER)
  // (OL_W_ID, OL_D_ID, OL_O_ID) Foreign Key, references (O_W_ID, O_D_ID,
  // O_ID)

  // (OL_SUPPLY_W_ID, OL_I_ID) Foreign Key, references (S_W_ID, S_I_ID)

  struct tpcc_order *r = new struct tpcc_order;
  struct tpcc_order_line *r_ol = new struct tpcc_order_line;
  struct tpcc_new_order *r_no = new struct tpcc_new_order;

  uint64_t *cperm = (uint64_t *)malloc(sizeof(uint64_t) * TPCC_NCUST_PER_DIST);
  assert(cperm);

  for (int d = 1; d <= TPCC_NDIST_PER_WH; d++) {
    init_permutation(&this->seed, cperm);

    for (int o = 1; o <= TPCC_NCUST_PER_DIST; o++) {
      uint32_t ckey = MAKE_CUST_KEY(w_id, d, o);

      // e = hash_insert(p, key, sizeof(struct tpcc_order), NULL);
      // assert(e);

      // p->ninserts++;
      // e->ref_count++;

      // r = (struct tpcc_order *)e->value;

      int c_id = cperm[o - 1];
      r->o_id = o;
      r->o_c_id = c_id;
      r->o_d_id = d;
      r->o_w_id = w_id;
      int o_entry = 2013;
      r->o_entry_d = 2013;
      if (o < 2101)
        r->o_carrier_id = URand(&this->seed, 1, 10);
      else
        r->o_carrier_id = 0;
      int o_ol_cnt = URand(&this->seed, 5, 15);
      r->o_ol_cnt = o_ol_cnt;
      r->o_all_local = 1;

      // insert order here
      void *hash_ptr = table_order->insertRecord(r, 0, 0);
      this->table_order->p_index->insert(ckey, hash_ptr);

      for (int ol = 1; ol <= o_ol_cnt; ol++) {
        uint64_t ol_pkey = MAKE_OL_KEY(w_id, d, o, ol);
        // hash_key ol_key = MAKE_HASH_KEY(ORDER_LINE_TID, ol_pkey);

        // struct elem *e_ol =
        //    hash_insert(p, ol_key, sizeof(struct tpcc_order_line), NULL);
        // assert(e_ol);

        // p->ninserts++;
        // e_ol->ref_count++;

        // struct tpcc_order_line *r_ol = (struct tpcc_order_line
        // *)e_ol->value;
        r_ol->ol_o_id = o;
        r_ol->ol_d_id = d;
        r_ol->ol_w_id = w_id;
        r_ol->ol_number = ol;
        r_ol->ol_i_id = URand(&this->seed, 1, 100000);
        r_ol->ol_supply_w_id = w_id;

        if (o < 2101) {
          r_ol->ol_delivery_d = o_entry;
          r_ol->ol_amount = 0;
        } else {
          r_ol->ol_delivery_d = 0;
          r_ol->ol_amount = (double)URand(&this->seed, 1, 999999) / 100;
        }
        r_ol->ol_quantity = 5;
        make_alpha_string(&this->seed, 24, 24, r_ol->ol_dist_info);

        //          uint64_t key = orderlineKey(wid, did, oid);
        //          index_insert(i_orderline, key, row, wh_to_part(wid));

        //          key = distKey(did, wid);
        //          index_insert(i_orderline_wd, key, row, wh_to_part(wid));

        // insert orderline here
        void *hash_ptr = table_order_line->insertRecord(r_ol, 0, 0);
        this->table_order_line->p_index->insert(ol_pkey, hash_ptr);
      }

      // NEW ORDER
      if (o > 2100) {
        // struct elem *e_no =
        //    hash_insert(p, key, sizeof(struct tpcc_new_order), NULL);
        // assert(e_no);

        // p->ninserts++;
        // e_no->ref_count++;

        // struct tpcc_new_order *r_no = (struct tpcc_new_order *)e_no->value;

        r_no->no_o_id = o;
        r_no->no_d_id = d;
        r_no->no_w_id = w_id;
        // insert new order here

        void *hash_ptr = table_new_order->insertRecord(r_no, 0, 0);
        this->table_new_order->p_index->insert(ckey, hash_ptr);
      }
    }
  }
  delete r;
  delete r_ol;
  delete r_no;
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

uint64_t TPCC::cust_derive_key(char *c_last, int c_d_id, int c_w_id) {
  uint64_t key = 0;
  char offset = 'A';
  for (uint32_t i = 0; i < strlen(c_last); i++)
    key = (key << 1) + (c_last[i] - offset);
  key = key << 10;
  key += c_w_id * TPCC_NDIST_PER_WH + c_d_id;

  return key;
}

void TPCC::load_customer(int w_id) {
  // Primary Key: (C_W_ID, C_D_ID, C_ID)
  // (C_W_ID, C_D_ID) Foreign Key, references (D_W_ID, D_ID)

  // void *hash_ptr = table_customer->insertRecord(r, 0, 0);
  // this->table_customer->p_index->insert(key, hash_ptr);

  struct tpcc_customer *r = new tpcc_customer;

  for (int d = 1; d <= TPCC_NDIST_PER_WH; d++) {
    for (int c = 1; c <= TPCC_NCUST_PER_DIST; c++) {
      uint32_t ckey = MAKE_CUST_KEY(w_id, d, c);

      // r = (struct tpcc_customer *)e->value;
      r->c_id = c;
      r->c_d_id = d;
      r->c_w_id = w_id;

      if (c <= 1000)
        set_last_name(c - 1, r->c_last);
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
      r->c_since = 0;
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
}

void TPCC::load_data(int num_threads) {
  // TODO: make it multi-threaded

  for (int w_id = 0; w_id < num_warehouse; w_id++) {
    load_stock(w_id);

    load_history(w_id);

    load_customer(w_id);

    load_order(w_id);
    if (w_id == 1) {
      load_item(w_id);  // ??
      // std::cout << "srv (%d): Loading item..\n", w_id);
    }

    load_warehouse(w_id);
    load_district(w_id);
  }
}

}  // namespace bench
