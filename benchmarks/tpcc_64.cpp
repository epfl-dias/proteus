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

#include "tpcc_64.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <locale>
#include <string>
#include <thread>

#include "utils/utils.hpp"

namespace bench {
std::mutex print_mutex;

#define extract_pid(v) CC_extract_pid(v)

inline date_t __attribute__((always_inline)) get_timestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

bool TPCC::exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                    ushort delta_ver, ushort partition_id) {
  struct tpcc_query *q = (struct tpcc_query *)stmts;
  switch (q->query_type) {
    case NEW_ORDER:
      return exec_neworder_txn(q, xid, partition_id, master_ver, delta_ver);
      break;
    // case PAYMENT:
    //   return exec_payment_txn(q, xid, partition_id, master_ver, delta_ver);
    //   break;
    // case ORDER_STATUS:
    //   return exec_orderstatus_txn(q, xid, partition_id, master_ver,
    //   delta_ver); break;
    // case DELIVERY:
    //   return exec_delivery_txn(q, xid, partition_id, master_ver, delta_ver);
    //   break;
    // case STOCK_LEVEL:
    //   return exec_stocklevel_txn(q, xid, partition_id, master_ver,
    //   delta_ver); break;
    default:
      assert(false);
      break;
  }
  return true;
}

void TPCC::gen_txn(int wid, void *q, ushort partition_id) {
  // mprotect(q, sizeof(struct tpcc_query), PROT_WRITE);
  static thread_local uint sequence_counter = 0;
  static thread_local TPCC_QUERY_TYPE *seq = sequence;

  switch (seq[sequence_counter++ % MIX_COUNT]) {
    case bench::NEW_ORDER:
      tpcc_get_next_neworder_query(wid, q);
      break;
    // case bench::PAYMENT:
    //   tpcc_get_next_payment_query(wid, q);
    //   break;
    // case bench::ORDER_STATUS:
    //   tpcc_get_next_orderstatus_query(wid, q);
    //   break;
    // case bench::DELIVERY:
    //   tpcc_get_next_delivery_query(wid, q);
    //   break;
    // case bench::STOCK_LEVEL:
    //   tpcc_get_next_stocklevel_query(wid, q);
    //   break;
    default: {
      std::unique_lock<std::mutex> lk(print_mutex);
      std::cout << "Unknown query type: "
                << sequence[sequence_counter++ % MIX_COUNT] << std::endl;

      assert(false);
      break;
    }
  }
  // mprotect(q, sizeof(struct tpcc_query), PROT_READ);
}

void TPCC::init_tpcc_seq_array() {
  int total = 0;
  for (int i = 0; i < NO_MIX; ++i) {
    sequence[i] = NEW_ORDER;
  }
  total = NO_MIX;
  for (int i = 0; i < P_MIX; ++i) {
    sequence[i + total] = PAYMENT;
  }
  total = total + P_MIX;
  for (int i = 0; i < OS_MIX; ++i) {
    sequence[i + total] = ORDER_STATUS;
  }
  total = total + OS_MIX;
  for (int i = 0; i < D_MIX; ++i) {
    sequence[i + total] = DELIVERY;
  }
  total = total + D_MIX;
  for (int i = 0; i < SL_MIX; ++i) {
    sequence[i + total] = STOCK_LEVEL;
  }
  total = total + SL_MIX;
  // shuffle elements of the sequence array
  srand(time(nullptr));
  for (int i = MIX_COUNT - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    TPCC_QUERY_TYPE temp = sequence[i];
    sequence[i] = sequence[j];
    sequence[j] = temp;
  }
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
    default:
      break;
  }
  std::cout << "\tw_id: " << q->w_id << std::endl;
  std::cout << "\td_id: " << q->d_id << std::endl;
  std::cout << "\tc_id: " << q->c_id << std::endl;
  std::cout << "\tol_cnt: " << q->ol_cnt << std::endl;

  std::cout << "-----------------------" << std::endl;
}

bool TPCC::exec_neworder_txn(const struct tpcc_query *q, uint64_t xid,
                             ushort partition_id, ushort master_ver,
                             ushort delta_ver) {
  // plus one for district
  global_conf::IndexVal *idx_w_locks[TPCC_MAX_OL_PER_ORDER + 1] = {};
  uint num_locks = 0;

  /*
   * EXEC SQL SELECT d_next_o_id, d_tax
   *  INTO :d_next_o_id, :d_tax
   *  FROM district WHERE d_id = :d_id AND d_w_id = :w_id;
   */

  // ACQUIRE WRITE_LOCK FOR DISTRICT
  idx_w_locks[num_locks] =
      (global_conf::IndexVal *)table_district->p_index->find(
          (uint64_t)MAKE_DIST_KEY(q->w_id, q->d_id));

  // uint64_t dt_k = MAKE_DIST_KEY(w_id, d_id);

  // if (dt_k / (18 * TPCC_NDIST_PER_WH) != partition_id) {
  //   std::unique_lock<std::mutex> lk(print_mutex);

  //   std::cout << "WID: " << w_id << std::endl;
  //   std::cout << "d_id: " << d_id << std::endl;
  //   std::cout << "d_key: " << dt_k << std::endl;
  //   std::cout << "partition_id: " << partition_id << std::endl;
  // }
  // assert(dt_k / (18 * TPCC_NDIST_PER_WH) == partition_id);

  bool e_false = false;
  assert(idx_w_locks[num_locks] != NULL || idx_w_locks[num_locks] != nullptr);
  if (idx_w_locks[num_locks]->write_lck.compare_exchange_strong(e_false,
                                                                true)) {
#if !tpcc_dist_txns
    assert(extract_pid(idx_w_locks[num_locks]->VID) == partition_id);
#endif
    num_locks++;
  } else {
    return false;
  }

  // ACQUIRE LOCK FOR STOCK
  for (int ol_number = 0; ol_number < q->ol_cnt; ol_number++) {
    uint64_t stock_key = MAKE_STOCK_KEY(q->item[ol_number].ol_supply_w_id,
                                        q->item[ol_number].ol_i_id);

    idx_w_locks[num_locks] =
        (global_conf::IndexVal *)table_stock->p_index->find(stock_key);

    assert(idx_w_locks[num_locks] != NULL || idx_w_locks[num_locks] != nullptr);
    bool e_false_s = false;
    if (idx_w_locks[num_locks]->write_lck.compare_exchange_strong(e_false_s,
                                                                  true)) {
#if !tpcc_dist_txns
      assert(extract_pid(idx_w_locks[num_locks]->VID) == partition_id);
#endif

      num_locks++;
    } else {
      txn::CC_MV2PL::release_locks(idx_w_locks, num_locks);
      return false;
    }
  }

  // Until this point, we have acquire all the necessary write locks,
  // we may now begin with reads and inserts which are never gonna abort in
  // MV2PL.

  /*
   * EXEC SQL SELECT s_quantity, s_d ata,
   * s_d ist_01, s_dist_02, s_d ist_03, s_d ist_04, s_d ist_05
   * s_d ist_06, s_dist_07, s_d ist_08, s_d ist_09, s_d ist_10
   * IN TO :s_quantity, :s_d ata,
   * :s_d ist_01, :s_d ist_02, :s_dist_03, :s_d ist_04, :s_d ist_05
   * :s_d ist_06, :s_d ist_07, :s_dist_08, :s_d ist_09, :s_d ist_10
   * FROM stock
   * WH ERE s_i_id = :ol_i_id AN D s_w _id = :ol_supply_w _id ;
   */

  struct __attribute__((packed)) st_read {
    int32_t s_quantity;
    uint32_t s_ytd;
    uint32_t s_order_cnt;
    uint32_t s_remote_cnt;
  };
  struct st_read st_rec = {};
  const ushort stock_col_rw[] = {2, TPCC_NDIST_PER_WH + 3,
                                 TPCC_NDIST_PER_WH + 4, TPCC_NDIST_PER_WH + 5};

  for (uint32_t ol_number = 0; ol_number < q->ol_cnt; ol_number++) {
    uint32_t ol_quantity = q->item[ol_number].ol_quantity;
    assert((ol_number + 1) < num_locks);

    global_conf::IndexVal *st_idx_ptr = idx_w_locks[ol_number + 1];
    assert(st_idx_ptr != NULL || st_idx_ptr != nullptr);

    st_idx_ptr->latch.acquire();

    if (txn::CC_MV2PL::is_readable(st_idx_ptr->t_min, xid)) {
      table_stock->getRecordByKey(st_idx_ptr->VID, stock_col_rw, 4, &st_rec);
    } else {
      struct tpcc_stock *s_r =
          (struct tpcc_stock *)st_idx_ptr->delta_ver->get_readable_ver(xid);

      st_rec.s_quantity = s_r->s_quantity;
      st_rec.s_ytd = s_r->s_ytd;
      st_rec.s_order_cnt = s_r->s_order_cnt;
      st_rec.s_remote_cnt = s_r->s_remote_cnt;
    }

    // NOW UPDATE
    // update = s_ytd, s_order_cnt, s_remote_cnt, s_quantity

    uint32_t s_quantity = st_rec.s_quantity;
    st_rec.s_ytd += ol_quantity;
    st_rec.s_order_cnt++;
    if (q->remote) {
      st_rec.s_remote_cnt++;
    }
    uint32_t quantity;
    if (s_quantity > ol_quantity + 10)
      quantity = s_quantity - ol_quantity;
    else
      quantity = s_quantity - ol_quantity + 91;

    st_rec.s_quantity = quantity;

    table_stock->updateRecord(st_idx_ptr, &st_rec, master_ver, delta_ver,
                              stock_col_rw, 4);
    st_idx_ptr->t_min = xid;
    // early release of locks?
    st_idx_ptr->latch.release();
  }

  /*
   * EXEC SQL UPDATE district SET d _next_o_id = :d _next_o_id + 1
   * WHERE d _id = :d_id AN D d _w _id = :w _id ;
   */

  const ushort dist_col_scan[] = {8, 10};
  const ushort dist_col_upd[] = {10};

  struct __attribute__((packed)) dist_read {
    double d_tax;
    uint64_t d_next_o_id;
  };
  struct dist_read dist_no_read = {};

  global_conf::IndexVal *d_idx_ptr = idx_w_locks[0];
  // assert(extract_pid(idx_w_locks[0]->VID) == partition_id);
  // assert(extract_pid(d_idx_ptr->VID) == partition_id);

  d_idx_ptr->latch.acquire();

  if (txn::CC_MV2PL::is_readable(d_idx_ptr->t_min, xid)) {
    table_district->getRecordByKey(d_idx_ptr->VID, dist_col_scan, 2,
                                   &dist_no_read);
  } else {
    struct tpcc_district *d_r =
        (struct tpcc_district *)d_idx_ptr->delta_ver->get_readable_ver(xid);

    dist_no_read.d_tax = d_r->d_tax;
    dist_no_read.d_next_o_id = d_r->d_next_o_id;
  }

  uint64_t d_next_o_id_upd = dist_no_read.d_next_o_id + 1;
  table_district->updateRecord(d_idx_ptr, &d_next_o_id_upd, master_ver,
                               delta_ver, dist_col_upd, 1);

  d_idx_ptr->t_min = xid;
  d_idx_ptr->latch.release();

  // TIME TO RELEASE LOCKS AS WE ARE NOT GONNA UPDATE ANYTHING
  txn::CC_MV2PL::release_locks(idx_w_locks, num_locks);

  // if (dist_no_read.d_next_o_id >= TPCC_MAX_ORD_PER_DIST) {
  //   std::unique_lock<std::mutex> lk(print_mutex);
  //   table_order->reportUsage();
  //   std::cout << "---Overflow" << std::endl;

  //   std::cout << "WID::" << q->w_id << std::endl;
  //   std::cout << "DID::" << q->d_id << std::endl;
  //   std::cout << "dist_no_read.d_next_o_id::" << dist_no_read.d_next_o_id
  //             << std::endl;
  //   std::cout << "PID::" << partition_id << std::endl;
  // }
  assert(dist_no_read.d_next_o_id < TPCC_MAX_ORD_PER_DIST - 1);

  /*
   * EXEC SQL SELECT c_discount, c_last, c_credit, w_tax
   * INTO :c_discount, :c_last, :c_credit, :w_tax
   * FROM customer, warehouse
   * WHERE w_id = :w_id AND c_w_id = w_id AND c_d_id = :d_id AND c_id =
   * :c_id;
   */

  double w_tax = 0.0;
  const ushort w_col_scan[] = {7};  // position in columns
  // std::cout << "WID::" << w_id << std::endl;
  global_conf::IndexVal *w_idx_ptr =
      (global_conf::IndexVal *)table_warehouse->p_index->find(
          (uint64_t)q->w_id);
#if !tpcc_dist_txns
  assert(extract_pid(w_idx_ptr->VID) == partition_id);
#endif

  w_idx_ptr->latch.acquire();
  if (txn::CC_MV2PL::is_readable(w_idx_ptr->t_min, xid)) {
    table_warehouse->getRecordByKey(w_idx_ptr->VID, w_col_scan, 1, &w_tax);
  } else {
    struct tpcc_warehouse *w_r =
        (struct tpcc_warehouse *)w_idx_ptr->delta_ver->get_readable_ver(xid);
    w_tax = w_r->w_tax;
  }

  w_idx_ptr->latch.release();

  // CUSTOMER

  // const ushort cust_col_scan[] = {5, 13, 15};
  const ushort cust_col_scan[] = {15};
  struct __attribute__((packed)) cust_read {
    // char c_last[LAST_NAME_LEN + 1];
    // char c_credit[2];
    double c_discount;
  };
  struct cust_read cust_no_read = {};

  global_conf::IndexVal *c_idx_ptr =
      (global_conf::IndexVal *)table_customer->p_index->find(
          (uint64_t)MAKE_CUST_KEY(q->w_id, q->d_id, q->c_id));
  assert(c_idx_ptr != NULL || c_idx_ptr != nullptr);

  c_idx_ptr->latch.acquire();

  // if (extract_pid(c_idx_ptr->VID) != partition_id) {
  //   std::unique_lock<std::mutex> lk(print_mutex);
  //   std::cout << "q->c_id: " << q->c_id << std::endl;
  //   std::cout << "q->w_id: " << q->w_id << std::endl;
  //   std::cout << "q->d_id: " << q->d_id << std::endl;

  //   std::cout << "partition_id: " << partition_id << std::endl;

  //   table_customer->reportUsage();
  // }

#if !tpcc_dist_txns
  assert(extract_pid(c_idx_ptr->VID) == partition_id);
#endif

  if (txn::CC_MV2PL::is_readable(c_idx_ptr->t_min, xid)) {
    table_customer->getRecordByKey(c_idx_ptr->VID, cust_col_scan, 1,
                                   &cust_no_read);

  } else {
    struct tpcc_customer *c_r =
        (struct tpcc_customer *)c_idx_ptr->delta_ver->get_readable_ver(xid);

    // memcpy(cust_no_read.c_last, c_r->c_last, LAST_NAME_LEN + 1);
    // memcpy(cust_no_read.c_credit, c_r->c_credit, 2);
    cust_no_read.c_discount = c_r->c_discount;
  }
  c_idx_ptr->latch.release();

  // INSERTS: Order Processing

  /*
   * EXEC SQL INSERT IN TO ORDERS (o_id , o_d _id , o_w _id , o_c_id ,
   * o_entry_d , o_ol_cnt, o_all_local)
   * VALUES (:o_id , :d _id , :w _id , :c_id ,
   * :d atetime, :o_ol_cnt, :o_all_local);
   */

  uint64_t order_key =
      MAKE_ORDER_KEY(q->w_id, q->d_id, dist_no_read.d_next_o_id);
  // assert(order_key < TPCC_MAX_ORDER_INITIAL_CAP);

  // if (order_key / 72000000 != partition_id) {
  //   std::unique_lock<std::mutex> lk(print_mutex);
  //   std::cout << "key: " << order_key << std::endl;
  //   std::cout << "q->w_id: " << q->w_id << std::endl;
  //   std::cout << "q->d_id: " << q->d_id << std::endl;
  //   std::cout << "dist_no_read.d_next_o_id: " << dist_no_read.d_next_o_id
  //             << std::endl;
  //   std::cout << "partition_id: " << partition_id << std::endl;
  //   std::cout << "pky: " << (order_key / 72000000) << std::endl;

  //   table_order->reportUsage();
  //   table_order_line->reportUsage();
  // }
  // assert(order_key / TPCC_MAX_ORDER_INITIAL_CAP_PER_PARTITION ==
  // partition_id);

  struct tpcc_order o_r = {};

  o_r.o_id = dist_no_read.d_next_o_id;
  o_r.o_c_id = q->c_id;
  o_r.o_d_id = q->d_id;
  o_r.o_w_id = q->w_id;
  o_r.o_entry_d = q->o_entry_d;
  o_r.o_ol_cnt = q->ol_cnt;
  o_r.o_all_local = !(q->remote);  // for now only local

  void *o_idx_ptr =
      table_order->insertRecord(&o_r, xid, partition_id, master_ver);

#if index_on_order_tbl
  table_order->p_index->insert(order_key, o_idx_ptr);
#endif

  /*
   * EXEC SQL INSERT IN TO NEW_ORDER (no_o_id , no_d_id , no_w _id )
   * VALUES (:o_id , :d _id , :w _id );
   */

  struct tpcc_new_order no_r = {};

  no_r.no_o_id = dist_no_read.d_next_o_id;
  no_r.no_d_id = q->d_id;
  no_r.no_w_id = q->w_id;

  void *no_idx_ptr =
      table_new_order->insertRecord(&no_r, xid, partition_id, master_ver);
#if index_on_order_tbl
  table_new_order->p_index->insert(order_key, no_idx_ptr);
#endif

  const ushort i_col_scan[] = {3};

#if batch_insert_no_ol
  // Row-store
  struct tpcc_order_line ol_ins_batch_row[TPCC_MAX_OL_PER_ORDER] = {};
  // Column-store optimization - layout_column_store
  struct tpcc_order_line_batch ol_ins_batch_col = {};

#else
  struct tpcc_order_line ol_ins;
#endif

  uint64_t ol_key_batch[TPCC_MAX_OL_PER_ORDER];

  for (int ol_number = 0; ol_number < q->ol_cnt; ol_number++) {
    /*
     * EXEC SQL SELECT i_price, i_name , i_data
     * INTO :i_price, :i_name, :i_data
     * FROM item WHERE i_id = ol_i_id
     */

    double i_price = 0.0;

#if REPLICATED_ITEM_TABLE

    global_conf::IndexVal *item_idx_ptr =
        (global_conf::IndexVal *)table_item[partition_id]->p_index->find(
            (uint64_t)(q->item[ol_number].ol_i_id));
    item_idx_ptr->latch.acquire();
    if (txn::CC_MV2PL::is_readable(item_idx_ptr->t_min, xid)) {
      table_item[partition_id]->getRecordByKey(item_idx_ptr->VID, i_col_scan, 1,
                                               &i_price);
    } else {
      std::cout << "not readable 5" << std::endl;
      struct tpcc_item *i_r =
          (struct tpcc_item *)item_idx_ptr->delta_ver->get_readable_ver(xid);
      i_price = i_r->i_price;
    }
    item_idx_ptr->latch.release();

#else

    global_conf::IndexVal *item_idx_ptr =
        (global_conf::IndexVal *)table_item->p_index->find(
            (uint64_t)(q->item[ol_number].ol_i_id));
    item_idx_ptr->latch.acquire();
    if (txn::CC_MV2PL::is_readable(item_idx_ptr->t_min, xid)) {
      table_item->getRecordByKey(item_idx_ptr->VID, i_col_scan, 1, &i_price);
    } else {
      std::cout << "not readable 5" << std::endl;
      struct tpcc_item *i_r =
          (struct tpcc_item *)item_idx_ptr->delta_ver->get_readable_ver(xid);
      i_price = i_r->i_price;
    }
    item_idx_ptr->latch.release();

#endif

    // char *i_name = i_r->i_name;
    // char *i_data = i_r->i_data;

    /*
     * EXEC SQL INSERT
     * INTO order_line(ol_o_id, ol_d_id, ol_w_id, ol_number,
     * ol_i_id, ol_supply_w_id,
     * ol_quantity, ol_amount, ol_dist_info)
     * VALUES(:o_id, :d_id, :w_id, :ol_number,
     * :ol_i_id, :ol_supply_w_id,
     * :ol_quantity, :ol_amount, :ol_dist_info);
     */
#if batch_insert_no_ol
    // ol_ins.ol_dist_info[24];  //
    if (layout_column_store) {
      ol_ins_batch_col.ol_o_id[ol_number] = dist_no_read.d_next_o_id;

      ol_ins_batch_col.ol_quantity[ol_number] = q->item[ol_number].ol_quantity;
      ol_ins_batch_col.ol_amount[ol_number] =
          q->item[ol_number].ol_quantity * i_price *
          (1 + w_tax + dist_no_read.d_tax) * (1 - cust_no_read.c_discount);
      ol_ins_batch_col.ol_d_id[ol_number] = q->d_id;
      ol_ins_batch_col.ol_w_id[ol_number] = q->w_id;
      ol_ins_batch_col.ol_number[ol_number] = ol_number;
      ol_ins_batch_col.ol_i_id[ol_number] = q->item[ol_number].ol_i_id;
      ol_ins_batch_col.ol_supply_w_id[ol_number] =
          q->item[ol_number].ol_supply_w_id;
      ol_ins_batch_col.ol_delivery_d[ol_number] = get_timestamp();

    } else {
      ol_ins_batch_row[ol_number].ol_o_id = dist_no_read.d_next_o_id;

      ol_ins_batch_row[ol_number].ol_quantity = q->item[ol_number].ol_quantity;
      ol_ins_batch_row[ol_number].ol_amount =
          q->item[ol_number].ol_quantity * i_price *
          (1 + w_tax + dist_no_read.d_tax) * (1 - cust_no_read.c_discount);
      ol_ins_batch_row[ol_number].ol_d_id = q->d_id;
      ol_ins_batch_row[ol_number].ol_w_id = q->w_id;
      ol_ins_batch_row[ol_number].ol_number = ol_number;
      ol_ins_batch_row[ol_number].ol_i_id = q->item[ol_number].ol_i_id;
      ol_ins_batch_row[ol_number].ol_supply_w_id =
          q->item[ol_number].ol_supply_w_id;
      ol_ins_batch_row[ol_number].ol_delivery_d = get_timestamp();

      // uint64_t ol_key =
      //     MAKE_OL_KEY(q->w_id, q->d_id, dist_no_read.d_next_o_id, ol_number);
      // void *ol_idx_ptr = table_order_line->insertRecord(
      //     &ol_ins, xid, partition_id, master_ver);
      // table_order_line->p_index->insert(ol_key, ol_idx_ptr);
    }

    ol_key_batch[ol_number] =
        MAKE_OL_KEY(q->w_id, q->d_id, dist_no_read.d_next_o_id, ol_number);
    // assert(ol_key_batch[ol_number] /
    // (TPCC_MAX_ORDER_INITIAL_CAP_PER_PARTITION *
    //                                   TPCC_MAX_OL_PER_ORDER) ==
    //        partition_id);

#else

    ol_ins.ol_o_id = dist_no_read.d_next_o_id;

    ol_ins.ol_quantity = q->item[ol_number].ol_quantity;
    ol_ins.ol_amount = q->item[ol_number].ol_quantity * i_price *
                       (1 + w_tax + dist_no_read.d_tax) *
                       (1 - cust_no_read.c_discount);
    ol_ins.ol_d_id = q->d_id;
    ol_ins.ol_w_id = q->w_id;
    ol_ins.ol_number = ol_number;
    ol_ins.ol_i_id = q->item[ol_number].ol_i_id;
    ol_ins.ol_supply_w_id = q->item[ol_number].ol_supply_w_id;
    ol_ins.ol_delivery_d = get_timestamp();

    uint64_t ol_key =
        MAKE_OL_KEY(q->w_id, q->d_id, dist_no_read.d_next_o_id, ol_number);

    // assert(ol_key / (TPCC_MAX_ORDER_INITIAL_CAP_PER_PARTITION *
    //                  TPCC_MAX_OL_PER_ORDER) ==
    //        partition_id);

    void *ol_idx_ptr =
        table_order_line->insertRecord(&ol_ins, xid, partition_id, master_ver);
#if index_on_order_tbl
    table_order_line->p_index->insert(ol_key, ol_idx_ptr);
#endif

#endif
  }

#if batch_insert_no_ol
  void *ol_ptr = &ol_ins_batch_col;
  if (!layout_column_store) {
    ol_ptr = ol_ins_batch_row;
  }

  global_conf::IndexVal *ol_idx_ptr_batch =
      (global_conf::IndexVal *)table_order_line->insertRecordBatch(
          ol_ptr, q->ol_cnt, TPCC_MAX_OL_PER_ORDER, xid, partition_id,
          master_ver);
#if index_on_order_tbl
  for (uint ol_number = 0; ol_number < q->ol_cnt; ol_number++) {
    void *pt = (void *)(ol_idx_ptr_batch + ol_number);
    table_order_line->p_index->insert(ol_key_batch[ol_number], pt);
  }
#endif
#endif

  return true;
}

inline void TPCC::tpcc_get_next_neworder_query(int wid, void *arg) {
  // mprotect(arg, sizeof(struct tpcc_query), PROT_WRITE);
  static thread_local unsigned int seed_t = this->seed;
  static thread_local unsigned int n_wh = this->num_warehouse;
  // static thread_local unsigned int n_actv_wh = this->num_active_workers;

#if PARTITION_LOCAL_ITEM_TABLE
  static thread_local uint64_t start = wid * (TPCC_MAX_ITEMS / n_wh);
  static thread_local uint64_t end = start + (TPCC_MAX_ITEMS / n_wh);
#endif

  int ol_cnt, dup;
  struct tpcc_query *q = (struct tpcc_query *)arg;

  q->query_type = NEW_ORDER;
  q->w_id = wid % n_wh;

  // q->w_id = URand(&p->seed, 1, g_nservers);
  // static thread_local uint did = 0;

  // q->d_id = (did++ % TPCC_NDIST_PER_WH);
  q->d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);
  q->c_id = NURand(&seed_t, 1023, 0, TPCC_NCUST_PER_DIST - 1);
  q->rbk = URand(&seed_t, 1, 100);
  q->ol_cnt = URand(&seed_t, 5, TPCC_MAX_OL_PER_ORDER);
  q->o_entry_d = get_timestamp();  // TODO: fixx
  q->remote = 0;

  ol_cnt = q->ol_cnt;
  assert(ol_cnt <= TPCC_MAX_OL_PER_ORDER);

  for (int o = 0; o < ol_cnt; o++) {
    struct item *i = &q->item[o];

    do {
#if PARTITION_LOCAL_ITEM_TABLE

      i->ol_i_id = NURand(&seed_t, 8191, start, end - 1);
#else
      i->ol_i_id = NURand(&seed_t, 8191, 0, TPCC_MAX_ITEMS - 1);
#endif

      // no duplicates
      dup = 0;
      for (int j = 0; j < o; j++)
        if (q->item[j].ol_i_id == i->ol_i_id) {
          dup = 1;
          break;
        }
    } while (dup);

#if tpcc_dist_txns

    int x = URand(&seed_t, 0, 100);
    if (g_dist_threshold == 100) x = 2;
    if (x > 1 || this->num_active_workers == 1) {
      i->ol_supply_w_id = wid;

    } else {
      while ((i->ol_supply_w_id =
                  URand(&seed_t, 0, this->num_active_workers - 1)) == q->w_id)
        ;

      q->remote = 1;
    }
#else
    i->ol_supply_w_id = wid;
#endif

    i->ol_quantity = URand(&seed_t, 1, 10);
  }

  // print_tpcc_query(arg);
  // mprotect(arg, sizeof(struct tpcc_query), PROT_READ);
}

TPCC::TPCC(std::string name, int num_warehouses, int active_warehouse,
           bool layout_column_store, int g_dist_threshold, std::string csv_path,
           bool is_ch_benchmark)
    : Benchmark(name, active_warehouse, std::thread::hardware_concurrency(),
                g_num_partitions),
      num_warehouse(num_warehouses),
      g_dist_threshold(g_dist_threshold),
      csv_path(csv_path),
      is_ch_benchmark(is_ch_benchmark),
      layout_column_store(layout_column_store) {
  this->schema = &storage::Schema::getInstance();
  this->seed = rand();

  uint64_t total_districts = TPCC_NDIST_PER_WH * (this->num_warehouse);
  uint64_t max_customers = TPCC_NCUST_PER_DIST * total_districts;
  uint64_t max_orders = TPCC_MAX_ORDER_INITIAL_CAP;

  max_orders = (max_orders / NUM_SOCKETS) * g_num_partitions;
  // std::cout << "MAX ORDERS: " << max_orders << std::endl;
  // std::cout << "g_num_partitions: " << g_num_partitions << std::endl;
  uint64_t max_order_line = TPCC_MAX_OL_PER_ORDER * max_orders;
  uint64_t max_stock = TPCC_MAX_ITEMS * (this->num_warehouse);

  std::vector<std::thread> loaders;

  loaders.emplace_back(
      [this]() { this->create_tbl_warehouse(this->num_warehouse); });
  loaders.emplace_back([this, total_districts]() {
    this->create_tbl_district(total_districts);
  });
  loaders.emplace_back(
      [this, max_customers]() { this->create_tbl_customer(max_customers); });
  loaders.emplace_back(
      [this, max_customers]() { this->create_tbl_history(max_customers); });
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

  std::cout << "Total Memory Reserved for Tables: "
            << (double)this->schema->total_mem_reserved / (1024 * 1024 * 1024)
            << " GB" << std::endl;
  std::cout << "Total Memory Reserved for Deltas: "
            << (double)this->schema->total_delta_mem_reserved /
                   (1024 * 1024 * 1024)
            << " GB" << std::endl;
  cust_sec_index = new indexes::HashIndex<uint64_t, struct secondary_record>();
  // cust_sec_index->reserve(max_customers);

  init_tpcc_seq_array();
}

void TPCC::create_tbl_warehouse(uint64_t num_warehouses) {
  // Primary Key: W_ID
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  struct tpcc_warehouse tmp;

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

  table_warehouse = schema->create_table(
      "tpcc_warehouse",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_warehouses);
}

void TPCC::create_tbl_district(uint64_t num_districts) {
  // Primary Key: (D_W_ID, D_ID) D_W_ID
  // Foreign Key, references W_ID
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  struct tpcc_district tmp;

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
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  struct tpcc_item tmp;

  columns.emplace_back("i_id", storage::INTEGER, sizeof(tmp.i_id));
  columns.emplace_back("i_im_id", storage::INTEGER, sizeof(tmp.i_im_id));
  columns.emplace_back("i_name", storage::VARCHAR, sizeof(tmp.i_name));
  columns.emplace_back("i_price", storage::FLOAT, sizeof(tmp.i_price));
  columns.emplace_back("i_data", storage::VARCHAR, sizeof(tmp.i_data));

#if REPLICATED_ITEM_TABLE
  for (int i = 0; i < g_num_partitions; i++) {
    table_item[i] = schema->create_table(
        "tpcc_item_" + std::to_string(i),
        (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
        columns, num_item, true, false, i);
  }
#else
  table_item = schema->create_table(
      "tpcc_item",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_item);
#endif
}

void TPCC::create_tbl_stock(uint64_t num_stock) {
  // Primary Key: (S_W_ID, S_I_ID)
  // S_W_ID Foreign Key, references W_ID
  // S_I_ID Foreign Key, references I_ID
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

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
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

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
      columns, num_history);
}

void TPCC::create_tbl_customer(uint64_t num_cust) {
  // Primary Key: (C_W_ID, C_D_ID, C_ID)
  // (C_W_ID, C_D_ID) Foreign Key, references (D_W_ID, D_ID)

  // TODO: get size of string vars from sizeof instead of hardcoded.

  struct tpcc_customer tmp;

  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

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

  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  columns.emplace_back("no_o_id", storage::INTEGER, sizeof(tmp.no_o_id));

  columns.emplace_back("no_d_id", storage::INTEGER, sizeof(tmp.no_d_id));
  columns.emplace_back("no_w_id", storage::INTEGER, sizeof(tmp.no_w_id));

  table_new_order = schema->create_table(
      "tpcc_new_order",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_new_order, index_on_order_tbl);
}

void TPCC::create_tbl_order(uint64_t num_order) {
  // Primary Key: (O_W_ID, O_D_ID, O_ID)
  // (O_W_ID, O_D_ID, O_C_ID) Foreign Key, references (C_W_ID, C_D_ID, C_ID)
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

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

  struct tpcc_order_line tmp;

  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

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
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  columns.emplace_back("su_suppkey", storage::INTEGER, sizeof(tmp.suppkey));

  columns.emplace_back("su_name", storage::STRING, sizeof(tmp.s_name));

  columns.emplace_back("su_address", storage::VARCHAR, sizeof(tmp.s_address));

  columns.emplace_back("su_nationkey", storage::INTEGER,
                       sizeof(tmp.s_nationkey));

  columns.emplace_back("su_phone", storage::STRING, sizeof(tmp.s_phone));

  columns.emplace_back("su_acctbal", storage::FLOAT, sizeof(tmp.s_acctbal));

  columns.emplace_back("su_comment", storage::VARCHAR, sizeof(tmp.s_comment));

  table_supplier = schema->create_table(
      "ch_supplier",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_supp);
}
void TPCC::create_tbl_region(uint64_t num_region) {
  // Primary Key: r_regionkey
  /*
     ushort r_regionkey;
      char r_name[12];      // var
      char r_comment[115];  // var
  */

  struct ch_region tmp;

  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  columns.emplace_back("r_regionkey", storage::INTEGER,
                       sizeof(tmp.r_regionkey));
  columns.emplace_back("r_name", storage::VARCHAR, sizeof(tmp.r_name));
  columns.emplace_back("r_comment", storage::VARCHAR, sizeof(tmp.r_comment));

  table_region = schema->create_table(
      "ch_region",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_region);
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
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  columns.emplace_back("n_nationkey", storage::INTEGER,
                       sizeof(tmp.n_nationkey));
  columns.emplace_back("n_name", storage::VARCHAR, sizeof(tmp.n_name));
  columns.emplace_back("n_regionkey", storage::INTEGER,
                       sizeof(tmp.n_regionkey));
  columns.emplace_back("n_comment", storage::VARCHAR, sizeof(tmp.n_comment));

  table_nation = schema->create_table(
      "ch_nation",
      (layout_column_store ? storage::COLUMN_STORE : storage::ROW_STORE),
      columns, num_nation);
}

/* A/C TPCC Specs*/
void TPCC::load_stock(int w_id, uint64_t xid, ushort partition_id,
                      ushort master_ver) {
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
void TPCC::load_item(int w_id, uint64_t xid, ushort partition_id,
                     ushort master_ver) {
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
#if REPLICATED_ITEM_TABLE

    for (int sc = 0; sc < g_num_partitions; sc++) {
      void *hash_ptr =
          table_item[sc]->insertRecord(&item_temp, xid, sc, master_ver);
      this->table_item[sc]->p_index->insert(key, hash_ptr);
    }

#else
    void *hash_ptr = table_item->insertRecord(
        &item_temp, 0, key / (TPCC_MAX_ITEMS / g_num_partitions), 0);
    this->table_item->p_index->insert(key, hash_ptr);
#endif
  }
}

/* A/C TPCC Specs*/
void TPCC::load_district(int w_id, uint64_t xid, ushort partition_id,
                         ushort master_ver) {
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
void TPCC::load_warehouse(int w_id, uint64_t xid, ushort partition_id,
                          ushort master_ver) {
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
  w_temp->w_ytd = 3000000.00;

  // txn_id = 0, master_ver = 0
  void *hash_ptr =
      table_warehouse->insertRecord(w_temp, xid, partition_id, master_ver);
  assert(hash_ptr != nullptr);
  this->table_warehouse->p_index->insert(w_id, hash_ptr);
  delete w_temp;
}

/* A/C TPCC Specs*/
void TPCC::load_history(int w_id, uint64_t xid, ushort partition_id,
                        ushort master_ver) {
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

/* A/C TPCC Specs*/
void TPCC::load_order(int w_id, uint64_t xid, ushort partition_id,
                      ushort master_ver) {
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

  if (SF != 0) {
    total_orderline_ins = 6001215 * SF;
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

  std::vector<std::thread> loaders;

  for (int d = 0; d < TPCC_NDIST_PER_WH; d++) {
    init_permutation(&this->seed, cperm);

    loaders.emplace_back([this, d, order_per_dist, cperm, pre_orders, w_id, xid,
                          partition_id, master_ver]() {
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

        int o_ol_cnt = URand(&this->seed, 5, 15);

        if (SF != 0) {
          o_ol_cnt = 15;
        }

        r.o_ol_cnt = o_ol_cnt;
        r.o_all_local = 1;

        // insert order here
        void *hash_ptr_o =
            table_order->insertRecord(&r, xid, partition_id, master_ver);
#if index_on_order_tbl
        assert(hash_ptr_o != nullptr || hash_ptr_o != NULL);
        this->table_order->p_index->insert(ckey, hash_ptr_o);
#endif

        for (int ol = 0; ol < o_ol_cnt; ol++) {
          struct tpcc_order_line r_ol = {};  // new struct tpcc_order_line;

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
            r_ol.ol_amount = (double)URand(&this->seed, 1, 999999) / 100;
          }
          r_ol.ol_quantity = 5;
          // make_alpha_string(&this->seed, 24, 24, r_ol.ol_dist_info);

          // insert orderline here
          void *hash_ptr_ol = table_order_line->insertRecord(
              &r_ol, xid, partition_id, master_ver);
#if index_on_order_tbl
          assert(hash_ptr_ol != nullptr || hash_ptr_ol != NULL);
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

          void *hash_ptr_no = table_new_order->insertRecord(
              &r_no, xid, partition_id, master_ver);
#if index_on_order_tbl
          assert(hash_ptr_no != nullptr || hash_ptr_no != NULL);
          this->table_new_order->p_index->insert(ckey, hash_ptr_no);
#endif
        }
      }
    });
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

uint64_t TPCC::cust_derive_key(char *c_last, int c_d_id, int c_w_id) {
  uint64_t key = 0;
  char offset = 'A';
  for (uint32_t i = 0; i < strlen(c_last); i++)
    key = (key << 1) + (c_last[i] - offset);
  key = key << 10;
  key += c_w_id * TPCC_NDIST_PER_WH + c_d_id;

  return key;
}

/* A/C TPCC Specs*/
void TPCC::load_customer(int w_id, uint64_t xid, ushort partition_id,
                         ushort master_ver) {
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

void TPCC::load_data(int num_threads) { assert(false && "Not Implemented"); }

void TPCC::pre_run(int wid, uint64_t xid, ushort partition_id,
                   ushort master_ver) {
  // static std::mutex print_mutex;
  // {
  //   std::unique_lock<std::mutex> lk(print_mutex);
  //   std::cout << "pre-run-------------------------------" << std::endl;
  //   std::cout << "pid: " << partition_id << std::endl;
  //   std::cout << "wid: " << wid << std::endl;
  // }
  if (wid >= this->num_warehouse) return;

  assert(partition_id < g_num_partitions);

#if REPLICATED_ITEM_TABLE
  if (wid == partition_id) load_item(wid, xid, partition_id, master_ver);

#else
  if (wid == 0) load_item(wid, xid, partition_id, master_ver);
#endif

  load_warehouse(wid, xid, partition_id, master_ver);

  load_district(wid, xid, partition_id, master_ver);

  load_stock(wid, xid, partition_id, master_ver);

  load_history(wid, xid, partition_id, master_ver);

  load_customer(wid, xid, partition_id, master_ver);

#if !debug_dont_load_order
  load_order(wid, xid, partition_id, master_ver);
#endif
}

}  // namespace bench
