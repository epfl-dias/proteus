/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <sys/mman.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <operators/relbuilder-factory.hpp>
#include <plan/catalog-parser.hpp>
#include <plan/prepared-statement.hpp>
#include <routing/degree-of-parallelism.hpp>
#include <string>

#include "aeolus-plugin.hpp"
#include "storage/column_store.hpp"
#include "threadpool/thread.hpp"
#include "tpcc_64.hpp"
#include "utils/utils.hpp"

namespace bench {

bool TPCC::exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                    ushort delta_ver, ushort partition_id) {
  struct tpcc_query *q = (struct tpcc_query *)stmts;
  switch (q->query_type) {
    case NEW_ORDER:
      return exec_neworder_txn(q, xid, partition_id, master_ver, delta_ver);
      break;
    case PAYMENT:
      return exec_payment_txn(q, xid, partition_id, master_ver, delta_ver);
      break;
    case ORDER_STATUS:
      return exec_orderstatus_txn(q, xid, partition_id, master_ver, delta_ver);
      break;
    case DELIVERY:
      return exec_delivery_txn(q, xid, partition_id, master_ver, delta_ver);
      break;
    case STOCK_LEVEL:
      return exec_stocklevel_txn(q, xid, partition_id, master_ver, delta_ver);
      break;
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
    case bench::PAYMENT:
      tpcc_get_next_payment_query(wid, q);
      break;
    case bench::ORDER_STATUS:
      tpcc_get_next_orderstatus_query(wid, q);
      break;
    case bench::DELIVERY:
      tpcc_get_next_delivery_query(wid, q);
      break;
    case bench::STOCK_LEVEL:
      tpcc_get_next_stocklevel_query(wid, q);
      break;
    default: {
      LOG(INFO) << "Unknown query type: "
                << sequence[sequence_counter++ % MIX_COUNT] << std::endl;

      assert(false);
      break;
    }
  }
  // mprotect(q, sizeof(struct tpcc_query), PROT_READ);
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
  if (idx_w_locks[num_locks]->write_lck.try_lock()) {
#if !tpcc_dist_txns
    assert(CC_extract_pid(idx_w_locks[num_locks]->VID) == partition_id);
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
    if (idx_w_locks[num_locks]->write_lck.try_lock()) {
#if !tpcc_dist_txns
      assert(CC_extract_pid(idx_w_locks[num_locks]->VID) == partition_id);
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
  // assert(CC_extract_pid(idx_w_locks[0]->VID) == partition_id);
  // assert(CC_extract_pid(d_idx_ptr->VID) == partition_id);

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
  assert(CC_extract_pid(w_idx_ptr->VID) == partition_id);
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

  // if (CC_extract_pid(c_idx_ptr->VID) != partition_id) {
  //   std::unique_lock<std::mutex> lk(print_mutex);
  //   std::cout << "q->c_id: " << q->c_id << std::endl;
  //   std::cout << "q->w_id: " << q->w_id << std::endl;
  //   std::cout << "q->d_id: " << q->d_id << std::endl;

  //   std::cout << "partition_id: " << partition_id << std::endl;

  //   table_customer->reportUsage();
  // }

#if !tpcc_dist_txns
  assert(CC_extract_pid(c_idx_ptr->VID) == partition_id);
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
  struct tpcc_orderline ol_ins_batch_row[TPCC_MAX_OL_PER_ORDER] = {};
  // Column-store optimization - layout_column_store
  struct tpcc_orderline_batch ol_ins_batch_col = {};

#else
  struct tpcc_orderline ol_ins = {};
#endif

  uint64_t ol_key_batch[TPCC_MAX_OL_PER_ORDER];

  for (int ol_number = 0; ol_number < q->ol_cnt; ol_number++) {
    /*
     * EXEC SQL SELECT i_price, i_name , i_data
     * INTO :i_price, :i_name, :i_data
     * FROM item WHERE i_id = ol_i_id
     */

    double i_price = 0.0;

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
      ol_ins_batch_col.ol_delivery_d[ol_number] = 0;  // get_timestamp();

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
      ol_ins_batch_row[ol_number].ol_delivery_d = 0;  // get_timestamp();

      // uint64_t ol_key =
      //     MAKE_OL_KEY(q->w_id, q->d_id, dist_no_read.d_next_o_id,
      //     ol_number);
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
    ol_ins.ol_delivery_d = 0;  // get_timestamp();

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

inline bool TPCC::exec_orderstatus_txn(struct tpcc_query *q, uint64_t xid,
                                       ushort partition_id, ushort master_ver,
                                       ushort delta_ver) {
  int ol_number = 1;

  uint32_t cust_id = std::numeric_limits<uint32_t>::max();

  if (q->by_last_name) {
    /*
      EXEC SQL SELECT count(c_id) INTO :namecnt
            FROM customer
            WHERE c_last=:c_last AND c_d_id=:d_id AND c_w_id=:w_id;

      EXEC SQL DECLARE c_name CURSOR FOR
          SELECT c_balance, c_first, c_middle, c_id
          FROM customer
          WHERE c_last=:c_last AND c_d_id=:d_id AND c_w_id=:w_id
          ORDER BY c_first;
      EXEC SQL OPEN c_name;

      if (namecnt%2) namecnt++; / / Locate midpoint customer

      for (n=0; n<namecnt/ 2; n++)
      {
        EXEC SQL FETCH c_name
        INTO :c_balance, :c_first, :c_middle, :c_id;
      }

      EXEC SQL CLOSE c_name;
     */

    struct secondary_record sr;

    if (!this->cust_sec_index->find(
            cust_derive_key(q->c_last, q->c_d_id, q->c_w_id), sr)) {
      assert(false && "read_txn aborted! how!");
      return false;
    }

    assert(sr.sr_idx < MAX_OPS_PER_QUERY);

    struct cust_read c_recs[MAX_OPS_PER_QUERY];

    uint nmatch = fetch_cust_records(sr, c_recs, xid);

    assert(nmatch > 0);

    // now sort based on first name and get middle element
    // XXX: Inefficient bubble sort for now. Also the strcmp below has a huge
    // overhead. We need some sorted secondary index structure
    for (int i = 0; i < nmatch; i++) {
      for (int j = i + 1; j < nmatch; j++) {
        if (strcmp(c_recs[i].c_first, c_recs[j].c_first) > 0) {
          struct cust_read tmp = c_recs[i];
          c_recs[i] = c_recs[j];
          c_recs[j] = tmp;
        }
      }
    }

    cust_id = MAKE_CUST_KEY(c_recs[nmatch / 2].c_w_id,
                            c_recs[nmatch / 2].c_d_id, c_recs[nmatch / 2].c_id);

  } else {  // by cust_id

    /*
     * EXEC SQL SELECT c_balance, c_first, c_middle, c_last
     * INTO :c_balance, :c_first, :c_middle, :c_last
     * FROM customer
     *  WHERE c_id=:c_id AND c_d_id=:d_id AND c_w_id=:w_id;
     */

    cust_id = MAKE_CUST_KEY(q->c_w_id, q->c_d_id, q->c_id);

    assert(cust_id != std::numeric_limits<uint32_t>::max());

    global_conf::IndexVal *c_idx_ptr =
        (global_conf::IndexVal *)table_customer->p_index->find(cust_id);

    assert(c_idx_ptr != nullptr || c_idx_ptr != NULL);

    struct __attribute__((packed)) cust_read_t {
      char c_first[FIRST_NAME_LEN + 1];
      char c_middle[2];
      char c_last[LAST_NAME_LEN + 1];
      double c_balance;
    };

    const ushort cust_col_scan_read[] = {3, 4, 5, 15};
    struct cust_read_t c_r;

    c_idx_ptr->latch.acquire();

    if (txn::CC_MV2PL::is_readable(c_idx_ptr->t_min, xid)) {
      table_customer->getRecordByKey(c_idx_ptr->VID, cust_col_scan_read, 4,
                                     &c_r);

    } else {
      struct tpcc_customer *c_rr =
          (struct tpcc_customer *)c_idx_ptr->delta_ver->get_readable_ver(xid);
      assert(c_rr != nullptr || c_rr != NULL);
    }
    c_idx_ptr->latch.release();

    /* EXEC SQL SELECT o_id, o_carrier_id, o_entry_d
     * INTO :o_id, :o_carrier_id, :entdate
     * FROM orders
     *  ORDER BY o_id DESC;
     */

    /*
     * EXEC SQL DECLARE c_line CURSOR FOR
     * SELECT ol_i_id, ol_supply_w_id, ol_quantity,
         ol_amount, ol_delivery_d
     * FROM order_line
     *  WHERE ol_o_id=:o_id AND ol_d_id=:d_id AND ol_w_id=:w_id;
     */

    int counter1 = 0;

    struct __attribute__((packed)) p_order_read {
      uint32_t o_id;
      uint32_t o_carrier_id;
      uint32_t entdate;
      uint32_t o_ol_cnt;
    } p_or[TPCC_NCUST_PER_DIST];

    struct __attribute__((packed)) p_orderline_read {
      uint32_t ol_o_id;
      uint32_t ol_supply_w_id;
      date_t ol_delivery_d;
      uint32_t ol_quantity;
      double ol_amount;
    } p_ol_r;

    const ushort o_col_scan[] = {0, 4, 5, 6};
    const ushort ol_col_scan[] = {0, 5, 6, 7, 8};

    for (int o = TPCC_NCUST_PER_DIST - 1, i = 0; o > 0; o--, i++) {
      global_conf::IndexVal *o_idx_ptr =
          (global_conf::IndexVal *)table_order->p_index->find(
              MAKE_ORDER_KEY(q->w_id, q->d_id, o));

      assert(o_idx_ptr != nullptr || o_idx_ptr != NULL);

      o_idx_ptr->latch.acquire();
      if (txn::CC_MV2PL::is_readable(o_idx_ptr->t_min, xid)) {
        table_order->getRecordByKey(o_idx_ptr->VID, o_col_scan, 4, &p_or[i]);
      } else {
        struct tpcc_order *c_r =
            (struct tpcc_order *)o_idx_ptr->delta_ver->get_readable_ver(xid);
        assert(c_r != nullptr || c_r != NULL);
        p_or[i].o_id = c_r->o_id;
        p_or[i].o_carrier_id = c_r->o_carrier_id;
        p_or[i].entdate = c_r->o_entry_d;
        p_or[i].o_ol_cnt = c_r->o_ol_cnt;
      }
      o_idx_ptr->latch.release();

      for (ushort ol_number = 0; ol_number < p_or[i].o_ol_cnt; ++ol_number) {
        global_conf::IndexVal *ol_idx_ptr =
            (global_conf::IndexVal *)table_order_line->p_index->find(
                MAKE_OL_KEY(q->w_id, q->d_id, p_or[i].o_id, ol_number));

        assert(ol_idx_ptr != nullptr || ol_idx_ptr != NULL);

        ol_idx_ptr->latch.acquire();
        if (txn::CC_MV2PL::is_readable(ol_idx_ptr->t_min, xid)) {
          table_order_line->getRecordByKey(ol_idx_ptr->VID,

                                           ol_col_scan, 5, &p_ol_r);
        } else {
          struct tpcc_orderline *ol_r =
              (struct tpcc_orderline *)ol_idx_ptr->delta_ver->get_readable_ver(
                  xid);
          assert(ol_r != nullptr || ol_r != NULL);

          p_ol_r.ol_o_id = ol_r->ol_o_id;
          p_ol_r.ol_supply_w_id = ol_r->ol_supply_w_id;
          p_ol_r.ol_delivery_d = ol_r->ol_delivery_d;
          p_ol_r.ol_quantity = ol_r->ol_quantity;
          p_ol_r.ol_amount = ol_r->ol_amount;
          //
        }
        ol_idx_ptr->latch.release();
      }
    }
  }

  return true;
}

inline bool TPCC::exec_stocklevel_txn(struct tpcc_query *q, uint64_t xid,
                                      ushort partition_id, ushort master_ver,
                                      ushort delta_ver) {
  /*
  This transaction accesses
  order_line,
  stock,
  district
  */

  /*
   * EXEC SQL SELECT d_next_o_id INTO :o_id
   * FROM district
   * WHERE d_w_id=:w_id AND d_id=:d_id;
   */

  global_conf::IndexVal *d_idx_ptr =
      (global_conf::IndexVal *)table_district->p_index->find(
          MAKE_DIST_KEY(q->w_id, q->d_id));

  assert(d_idx_ptr != nullptr || d_idx_ptr != NULL);

  const ushort dist_col_scan_read[] = {10};
  uint32_t o_id = 0;

  d_idx_ptr->latch.acquire();

  if (txn::CC_MV2PL::is_readable(d_idx_ptr->t_min, xid)) {
    table_district->getRecordByKey(d_idx_ptr->VID, dist_col_scan_read, 1,
                                   &o_id);

  } else {
    struct tpcc_district *d_r =
        (struct tpcc_district *)d_idx_ptr->delta_ver->get_readable_ver(xid);
    assert(d_r != nullptr || d_r != NULL);

    o_id = d_r->d_next_o_id;
  }
  d_idx_ptr->latch.release();

  /*
   * EXEC SQL SELECT COUNT(DISTINCT (s_i_id)) INTO :stock_count
   * FROM order_line, stock
   * WHERE ol_w_id=:w_id AND
   * ol_d_id=:d_id AND ol_o_id<:o_id AND
   * ol_o_id>=:o_id-20 AND s_w_id=:w_id AND
   * s_i_id=ol_i_id AND s_quantity < :threshold;
   */

  uint32_t ol_o_id = o_id - 20;
  uint32_t ol_number = -1;
  uint32_t stock_count = 0;

  const ushort ol_col_scan_read[] = {4};
  const ushort st_col_scan_read[] = {2};

  while (ol_o_id < o_id) {
    while (ol_number < TPCC_MAX_OL_PER_ORDER) {
      ol_number++;

      // orderline first
      global_conf::IndexVal *ol_idx_ptr =
          (global_conf::IndexVal *)table_order_line->p_index->find(
              MAKE_OL_KEY(q->w_id, q->d_id, ol_o_id, ol_number));

      if (ol_idx_ptr == nullptr) continue;

      uint32_t ol_i_id;
      int32_t s_quantity;

      ol_idx_ptr->latch.acquire();

      if (txn::CC_MV2PL::is_readable(ol_idx_ptr->t_min, xid)) {
        table_order_line->getRecordByKey(ol_idx_ptr->VID, ol_col_scan_read, 1,
                                         &ol_i_id);

      } else {
        struct tpcc_orderline *ol_r =
            (struct tpcc_orderline *)ol_idx_ptr->delta_ver->get_readable_ver(
                xid);
        assert(ol_r != nullptr || ol_r != NULL);
        ol_i_id = ol_r->ol_i_id;
      }
      ol_idx_ptr->latch.release();

      // stock
      global_conf::IndexVal *st_idx_ptr =
          (global_conf::IndexVal *)table_stock->p_index->find(
              MAKE_STOCK_KEY(q->w_id, ol_i_id));

      assert(st_idx_ptr != nullptr || st_idx_ptr != NULL);

      st_idx_ptr->latch.acquire();

      if (txn::CC_MV2PL::is_readable(st_idx_ptr->t_min, xid)) {
        table_stock->getRecordByKey(st_idx_ptr->VID, st_col_scan_read, 1,
                                    &s_quantity);

      } else {
        struct tpcc_stock *st_r =
            (struct tpcc_stock *)st_idx_ptr->delta_ver->get_readable_ver(xid);
        assert(st_r != nullptr || st_r != NULL);
        s_quantity = st_r->s_quantity;
      }
      st_idx_ptr->latch.release();

      if (s_quantity < q->threshold) {
        stock_count++;
      }
    }
    ol_o_id = ol_o_id + 1;
  }

  return true;
}

inline uint TPCC::fetch_cust_records(const struct secondary_record &sr,
                                     struct cust_read *c_recs, uint64_t xid) {
  uint nmatch = 0;

  // struct cust_read {
  //   uint32_t c_id;
  //   ushort c_d_id;
  //   ushort c_w_id;
  //   char c_first[FIRST_NAME_LEN + 1];
  //   char c_last[LAST_NAME_LEN + 1];
  // };

  const ushort c_col_scan[] = {0, 1, 2, 3, 5};  // position in columns

  for (int i = 0; i < sr.sr_nids; i++) {
    global_conf::IndexVal *c_idx_ptr =
        (global_conf::IndexVal *)table_customer->p_index->find(
            (uint64_t)(sr.sr_rids[i]));

    assert(c_idx_ptr != nullptr || c_idx_ptr != NULL);

    c_idx_ptr->latch.acquire();

    if (txn::CC_MV2PL::is_readable(c_idx_ptr->t_min, xid)) {
      table_customer->getRecordByKey(c_idx_ptr->VID, c_col_scan, 5,
                                     &c_recs[nmatch]);
      c_idx_ptr->latch.release();
    } else {
      struct tpcc_customer *c_r =
          (struct tpcc_customer *)c_idx_ptr->delta_ver->get_readable_ver(xid);
      assert(c_r != nullptr || c_r != NULL);
      c_idx_ptr->latch.release();

      strcpy(c_recs[nmatch].c_first, c_r->c_first);
      strcpy(c_recs[nmatch].c_last, c_r->c_last);
      c_recs[nmatch].c_id = c_r->c_id;
      c_recs[nmatch].c_d_id = c_r->c_d_id;
      c_recs[nmatch].c_w_id = c_r->c_w_id;
    }

    nmatch++;
  }

  return nmatch;
}

inline bool TPCC::exec_payment_txn(struct tpcc_query *q, uint64_t xid,
                                   ushort partition_id, ushort master_ver,
                                   ushort delta_ver) {
  // // Updates..

  // // update warehouse (ytd) / district (ytd),
  // // update customer (balance/c_data)
  // // insert history

  // /*====================================================+
  //     Acquire Locks for Warehouse, District, Customer
  // +====================================================*/

  // Lock Warehouse
  global_conf::IndexVal *w_idx_ptr =
      (global_conf::IndexVal *)table_warehouse->p_index->find(q->w_id);

  assert(w_idx_ptr != NULL || w_idx_ptr != nullptr);
  bool e_false = false;
  if (!(w_idx_ptr->write_lck.try_lock())) {
    return false;
  }

  // Lock District
  global_conf::IndexVal *d_idx_ptr =
      (global_conf::IndexVal *)table_district->p_index->find(
          MAKE_DIST_KEY(q->w_id, q->d_id));

  assert(d_idx_ptr != NULL || d_idx_ptr != nullptr);
  e_false = false;
  if (!(d_idx_ptr->write_lck.try_lock())) {
    w_idx_ptr->write_lck.unlock();
    return false;
  }

  // Lock Customer

  // some logic is required here as customer can be by name or by id..

  uint32_t cust_id = std::numeric_limits<uint32_t>::max();

  if (q->by_last_name) {
    /*==========================================================+
      EXEC SQL SELECT count(c_id) INTO :namecnt
      FROM customer
      WHERE c_last=:c_last AND c_d_id=:c_d_id AND c_w_id=:c_w_id;
      +==========================================================*/

    struct secondary_record sr;

    if (!this->cust_sec_index->find(
            cust_derive_key(q->c_last, q->c_d_id, q->c_w_id), sr)) {
      // ABORT

      w_idx_ptr->write_lck.unlock();
      d_idx_ptr->write_lck.unlock();

      return false;
    }

    assert(sr.sr_idx < MAX_OPS_PER_QUERY);

    struct cust_read c_recs[MAX_OPS_PER_QUERY];

    uint nmatch = fetch_cust_records(sr, c_recs, xid);

    assert(nmatch > 0);

    /*============================================================================+
        for (n=0; n<namecnt/2; n++) {
            EXEC SQL FETCH c_byname
            INTO :c_first, :c_middle, :c_id,
                 :c_street_1, :c_street_2, :c_city, :c_state, :c_zip,
                 :c_phone, :c_credit, :c_credit_lim, :c_discount, :c_balance,
    :c_since;
            }
        EXEC SQL CLOSE c_byname;
    +=============================================================================*/
    // now sort based on first name and get middle element
    // XXX: Inefficient bubble sort for now. Also the strcmp below has a huge
    // overhead. We need some sorted secondary index structure
    for (int i = 0; i < nmatch; i++) {
      for (int j = i + 1; j < nmatch; j++) {
        if (strcmp(c_recs[i].c_first, c_recs[j].c_first) > 0) {
          struct cust_read tmp = c_recs[i];
          c_recs[i] = c_recs[j];
          c_recs[j] = tmp;
        }
      }
    }

    cust_id = MAKE_CUST_KEY(c_recs[nmatch / 2].c_w_id,
                            c_recs[nmatch / 2].c_d_id, c_recs[nmatch / 2].c_id);

  } else {  // by cust_id
    cust_id = MAKE_CUST_KEY(q->c_w_id, q->c_d_id, q->c_id);
  }

  assert(cust_id != std::numeric_limits<uint32_t>::max());

  global_conf::IndexVal *c_idx_ptr =
      (global_conf::IndexVal *)table_customer->p_index->find(cust_id);

  assert(c_idx_ptr != NULL || c_idx_ptr != nullptr);
  e_false = false;
  if (!(c_idx_ptr->write_lck.try_lock())) {
    w_idx_ptr->write_lck.unlock();
    d_idx_ptr->write_lck.unlock();
    return false;
  }

  // -------------  ALL LOCKS ACQUIRED

  /*====================================================+
      EXEC SQL UPDATE warehouse SET w_ytd = w_ytd + :h_amount
      WHERE w_id=:w_id;
  +====================================================*/
  /*===================================================================+
      EXEC SQL SELECT w_street_1, w_street_2, w_city, w_state, w_zip, w_name
      INTO :w_street_1, :w_street_2, :w_city, :w_state, :w_zip, :w_name
      FROM warehouse
      WHERE w_id=:w_id;
  +===================================================================*/

  // update warehouse and then release the write lock

  double w_ytd = 0.0;
  const ushort wh_col_scan_upd[] = {7};
  w_idx_ptr->latch.acquire();

  if (txn::CC_MV2PL::is_readable(w_idx_ptr->t_min, xid)) {
    table_warehouse->getRecordByKey(w_idx_ptr->VID, wh_col_scan_upd, 1, &w_ytd);
  } else {
    struct tpcc_warehouse *w_r =
        (struct tpcc_warehouse *)w_idx_ptr->delta_ver->get_readable_ver(xid);
    assert(w_r != nullptr || w_r != NULL);
    w_ytd = w_r->w_ytd;
  }

  w_ytd += q->h_amount;
  table_warehouse->updateRecord(w_idx_ptr, &w_ytd, master_ver, delta_ver,
                                wh_col_scan_upd, 1);

  w_idx_ptr->t_min = xid;
  w_idx_ptr->latch.release();
  w_idx_ptr->write_lck.unlock();

  /*=====================================================+
      EXEC SQL UPDATE district SET d_ytd = d_ytd + :h_amount
      WHERE d_w_id=:w_id AND d_id=:d_id;
  =====================================================*/
  /*====================================================================+
      EXEC SQL SELECT d_street_1, d_street_2, d_city, d_state, d_zip, d_name
      INTO :d_street_1, :d_street_2, :d_city, :d_state, :d_zip, :d_name
      FROM district
      WHERE d_w_id=:w_id AND d_id=:d_id;
  +====================================================================*/

  double d_ytd = 0.0;
  const ushort d_col_scan_upd[] = {8};
  d_idx_ptr->latch.acquire();

  if (txn::CC_MV2PL::is_readable(d_idx_ptr->t_min, xid)) {
    table_district->getRecordByKey(d_idx_ptr->VID, d_col_scan_upd, 1, &d_ytd);
  } else {
    struct tpcc_district *d_r =
        (struct tpcc_district *)d_idx_ptr->delta_ver->get_readable_ver(xid);
    assert(d_r != nullptr || d_r != NULL);
    d_ytd = d_r->d_ytd;
  }

  d_ytd += q->h_amount;

  table_district->updateRecord(d_idx_ptr, &d_ytd, master_ver, delta_ver,
                               d_col_scan_upd, 1);

  d_idx_ptr->t_min = xid;
  d_idx_ptr->latch.release();
  d_idx_ptr->write_lck.unlock();

  //---

  /*======================================================================+
    EXEC SQL UPDATE customer SET c_balance = :c_balance, c_data = :c_new_data
    WHERE c_w_id = :c_w_id AND c_d_id = :c_d_id AND c_id = :c_id;
    +======================================================================*/

  struct __attribute__((packed)) cust_rw_t {
    double c_balance;
    double c_ytd_payment;
    uint32_t c_payment_cnt;
    char c_data[501];
    char c_credit[2];
  };

  const ushort cust_col_scan_read[] = {15, 16, 17, 19, 12};
  ushort num_col_upd = 3;

  struct cust_rw_t cust_rw = {};

  c_idx_ptr->latch.acquire();

  if (txn::CC_MV2PL::is_readable(c_idx_ptr->t_min, xid)) {
    table_customer->getRecordByKey(c_idx_ptr->VID, cust_col_scan_read, 5,
                                   &cust_rw);

    cust_rw.c_balance -= q->h_amount;
    cust_rw.c_ytd_payment += q->h_amount;
    cust_rw.c_payment_cnt += 1;

  } else {
    struct tpcc_customer *c_rr =
        (struct tpcc_customer *)c_idx_ptr->delta_ver->get_readable_ver(xid);
    assert(c_rr != nullptr || c_rr != NULL);

    cust_rw.c_balance = c_rr->c_balance - q->h_amount;
    cust_rw.c_ytd_payment = c_rr->c_ytd_payment + q->h_amount;
    cust_rw.c_payment_cnt = c_rr->c_payment_cnt + 1;

    strcpy(cust_rw.c_data, c_rr->c_data);
  }

  if (cust_rw.c_credit[0] == 'B' && cust_rw.c_credit[1] == 'C') {
    num_col_upd++;

    /*=====================================================+
      EXEC SQL SELECT c_data
            INTO :c_data
            FROM customer
            WHERE c_w_id=:c_w_id AND c_d_id=:c_d_id AND c_id=:c_id;
            +=====================================================*/
    // char c_new_data[501];
    sprintf(cust_rw.c_data, "| %4d %2d %4d %2d %4d $%7.2f", q->c_id, q->c_d_id,
            q->c_w_id, q->d_id, q->w_id, q->h_amount);
    strncat(cust_rw.c_data, cust_rw.c_data, 500 - strlen(cust_rw.c_data));
  }

  table_customer->updateRecord(c_idx_ptr, &cust_rw, master_ver, delta_ver,
                               cust_col_scan_read, num_col_upd);

  c_idx_ptr->t_min = xid;
  c_idx_ptr->latch.release();
  c_idx_ptr->write_lck.unlock();

  /*
      char h_data[25];
      char * w_name = r_wh_local->get_value("W_NAME");
      char * d_name = r_dist_local->get_value("D_NAME");
      strncpy(h_data, w_name, 10);
      int length = strlen(h_data);
      if (length > 10) length = 10;
      strcpy(&h_data[length], "    ");
      strncpy(&h_data[length + 4], d_name, 10);
      h_data[length+14] = '\0';
  */

  /*=============================================================================+
    EXEC SQL INSERT INTO
    history (h_c_d_id, h_c_w_id, h_c_id, h_d_id, h_w_id, h_date, h_amount,
    h_data) VALUES (:c_d_id, :c_w_id, :c_id, :d_id, :w_id, :datetime,
    :h_amount, :h_data);
    +=============================================================================*/

  struct tpcc_history h_ins;
  h_ins.h_c_id = q->c_id;
  h_ins.h_c_d_id = q->c_d_id;
  h_ins.h_c_w_id = q->c_w_id;
  h_ins.h_d_id = q->d_id;
  h_ins.h_w_id = q->w_id;
  h_ins.h_date = get_timestamp();
  h_ins.h_amount = q->h_amount;

  void *hist_idx_ptr =
      table_history->insertRecord(&h_ins, xid, partition_id, master_ver);

  return true;
}

inline bool TPCC::exec_delivery_txn(struct tpcc_query *q, uint64_t xid,
                                    ushort partition_id, ushort master_ver,
                                    ushort delta_ver) {
  static thread_local uint64_t delivered_orders[TPCC_NDIST_PER_WH] = {0};

  date_t delivery_d = get_timestamp();

  for (int d_id = 0; d_id < TPCC_NDIST_PER_WH; d_id++) {
    const uint64_t order_end_idx =
        MAKE_ORDER_KEY(q->w_id, d_id, TPCC_MAX_ORD_PER_DIST);

    if (delivered_orders[d_id] == 0) {
      delivered_orders[d_id] = MAKE_ORDER_KEY(q->w_id, d_id, 0);
    }

    for (size_t o = delivered_orders[d_id]; o <= order_end_idx; o++) {
      global_conf::IndexVal *idx_w_locks[TPCC_MAX_OL_PER_ORDER + 2] = {};
      uint num_locks = 0;

      uint64_t okey = MAKE_ORDER_KEY(q->w_id, d_id, 0);

      // get_new_order
      struct tpcc_new_order no_read = {};

      global_conf::IndexVal *no_idx_ptr =
          (global_conf::IndexVal *)table_new_order->p_index->find(okey);
      if (no_idx_ptr == nullptr) {
        // LOG(INFO) << "W_ID: " << q->w_id << " | D_ID: " << d_id
        //           << " | order: " << o << "  .. DONE-S";
        break;
      }

      no_idx_ptr->latch.acquire();

      if (txn::CC_MV2PL::is_readable(no_idx_ptr->t_min, xid)) {
        table_new_order->getRecordByKey(no_idx_ptr->VID, nullptr, 0, &no_read);

      } else {
        // Actually this is the breaking condition of loop.
        std::cout << "DONE." << std::endl;
        LOG(INFO) << "W_ID: " << q->w_id << " | D_ID: " << d_id
                  << " | order: " << o << "  .. DONE";

        break;
      }
      no_idx_ptr->latch.release();

      // Get Order

      struct __attribute__((packed)) order_read_st {
        uint32_t o_c_id;
        uint32_t o_ol_cnt;
      };
      struct order_read_st order_read = {};

      order_read.o_c_id = std::numeric_limits<uint32_t>::max();

      const ushort order_col_scan[] = {3, 6};

      global_conf::IndexVal *o_idx_ptr =
          (global_conf::IndexVal *)table_order->p_index->find(okey);
      assert(o_idx_ptr != NULL || o_idx_ptr != nullptr);

      o_idx_ptr->latch.acquire();

      if (txn::CC_MV2PL::is_readable(o_idx_ptr->t_min, xid)) {
        table_order->getRecordByKey(o_idx_ptr->VID, order_col_scan, 2,
                                    &order_read);

      } else {
        assert(false && "impossible");
      }
      o_idx_ptr->latch.release();

      assert(order_read.o_c_id != std::numeric_limits<uint32_t>::max());
      assert(order_read.o_ol_cnt != 0);

      // lock on order, orderline and customer

      // ACQUIRE WRITE_LOCK FOR CUSTOMER
      idx_w_locks[num_locks] =
          (global_conf::IndexVal *)table_customer->p_index->find(
              (uint64_t)MAKE_CUST_KEY(q->w_id, d_id, order_read.o_c_id));

      bool e_false = false;
      assert(idx_w_locks[num_locks] != NULL ||
             idx_w_locks[num_locks] != nullptr);
      if (idx_w_locks[num_locks]->write_lck.try_lock()) {
#if !tpcc_dist_txns
        assert(CC_extract_pid(idx_w_locks[num_locks]->VID) == partition_id);
#endif
        num_locks++;
      } else {
        assert(false && "Not possible");
        return false;
      }

      // ACQUIRE WRITE_LOCK FOR ORDER
      idx_w_locks[num_locks] =
          (global_conf::IndexVal *)table_order->p_index->find(okey);

      e_false = false;
      assert(idx_w_locks[num_locks] != NULL ||
             idx_w_locks[num_locks] != nullptr);
      if (idx_w_locks[num_locks]->write_lck.try_lock()) {
#if !tpcc_dist_txns
        assert(CC_extract_pid(idx_w_locks[num_locks]->VID) == partition_id);
#endif
        num_locks++;
      } else {
        assert(false && "Not possible");
        return false;
      }

      // ACQUIRE WRITE_LOCK FOR ORDER-LINE
      for (uint ol_number = 0; ol_number < order_read.o_ol_cnt; ol_number++) {
        idx_w_locks[num_locks] =
            (global_conf::IndexVal *)table_order_line->p_index->find(
                (uint64_t)MAKE_OL_KEY(q->w_id, d_id, okey, ol_number));

        assert(idx_w_locks[num_locks] != NULL ||
               idx_w_locks[num_locks] != nullptr);
        bool e_false_s = false;
        if (idx_w_locks[num_locks]->write_lck.try_lock()) {
#if !tpcc_dist_txns
          assert(CC_extract_pid(idx_w_locks[num_locks]->VID) == partition_id);
#endif
          num_locks++;
        } else {
          txn::CC_MV2PL::release_locks(idx_w_locks, num_locks);
          assert(false && "Not possible");
          return false;
        }
      }

      // update order
      constexpr ushort o_col_upd[] = {5};
      o_idx_ptr->latch.acquire();
      table_order->updateRecord(o_idx_ptr, &q->o_carrier_id, master_ver,
                                delta_ver, o_col_upd, 1);
      o_idx_ptr->t_min = xid;
      o_idx_ptr->latch.release();

      // update orderline
      //    upd delivery_d and read ol_amount

      constexpr ushort ol_col_r[] = {8};
      constexpr ushort ol_col_upd[] = {6};

      double ol_amount_sum = 0;

      for (uint ol_number = 0; ol_number < order_read.o_ol_cnt; ol_number++) {
        global_conf::IndexVal *ol_idx_ptr = idx_w_locks[ol_number + 2];
        assert(ol_idx_ptr != NULL || ol_idx_ptr != nullptr);

        ol_idx_ptr->latch.acquire();

        if (txn::CC_MV2PL::is_readable(ol_idx_ptr->t_min, xid)) {
          double ol_amount = 0;
          table_order_line->getRecordByKey(ol_idx_ptr->VID, ol_col_r, 1,
                                           &ol_amount);
          ol_amount_sum += ol_amount;
        } else {
          assert(false && "Not possible");
        }

        table_order_line->updateRecord(ol_idx_ptr, &delivery_d, master_ver,
                                       delta_ver, ol_col_upd, 1);
        ol_idx_ptr->t_min = xid;
        ol_idx_ptr->latch.release();
      }

      // update customer

      constexpr ushort c_col_rw[] = {15};
      double c_balance = 0;
      global_conf::IndexVal *c_idx_ptr = idx_w_locks[0];
      assert(c_idx_ptr != NULL || c_idx_ptr != nullptr);

      c_idx_ptr->latch.acquire();

      if (txn::CC_MV2PL::is_readable(c_idx_ptr->t_min, xid)) {
        double ol_amount = 0;
        table_customer->getRecordByKey(c_idx_ptr->VID, c_col_rw, 1, &c_balance);
        c_balance += ol_amount_sum;
      } else {
        assert(false && "Not possible");
      }

      table_customer->updateRecord(c_idx_ptr, &c_balance, master_ver, delta_ver,
                                   c_col_rw, 1);
      c_idx_ptr->t_min = xid;
      c_idx_ptr->latch.release();

      // finally
      txn::CC_MV2PL::release_locks(idx_w_locks, num_locks);
      delivered_orders[d_id] = o;
    }
  }
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

inline void TPCC::tpcc_get_next_orderstatus_query(int wid, void *arg) {
  static thread_local unsigned int seed_t = this->seed;
  struct tpcc_query *q = (struct tpcc_query *)arg;
  q->query_type = ORDER_STATUS;
  q->w_id = wid;
  q->d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);
  q->c_w_id = wid;

#if tpcc_cust_sec_idx
  int y = URand(&seed_t, 1, 100);
  if (y <= 60) {
    // by last name
    q->by_last_name = TRUE;
    set_last_name(NURand(&seed_t, 255, 0, 999), q->c_last);
  } else {
    // by cust id
    q->by_last_name = FALSE;
    q->c_id = NURand(&seed_t, 1023, 0, TPCC_NCUST_PER_DIST - 1);
  }
#else
  q->by_last_name = FALSE;
  q->c_id = NURand(&seed_t, 1023, 0, TPCC_NCUST_PER_DIST - 1);
#endif
}

inline void TPCC::tpcc_get_next_payment_query(int wid, void *arg) {
  static thread_local unsigned int seed_t = this->seed;
  struct tpcc_query *q = (struct tpcc_query *)arg;
  q->query_type = PAYMENT;
  q->w_id = wid;
  q->d_w_id = wid;

  q->d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);
  q->h_amount = URand(&seed_t, 1, 5000);
  int x = URand(&seed_t, 1, 100);

#if tpcc_dist_txns
  if (x <= 85 || g_dist_threshold == 100) {
    // home warehouse
    q->c_d_id = q->d_id;
    q->c_w_id = wid;

  } else {
    q->c_d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);

    // remote warehouse if we have >1 wh
    if (this->num_active_workers > 1) {
      while ((q->c_w_id = URand(&seed_t, 0, this->num_active_workers - 1)) ==
             wid)
        ;

    } else {
      q->c_w_id = wid;
    }
  }

#else
  q->c_d_id = q->d_id;
  q->c_w_id = wid;
#endif

#if tpcc_cust_sec_idx
  int y = URand(&seed_t, 1, 100);
  if (y <= 60) {
    // by last name
    q->by_last_name = TRUE;
    set_last_name(NURand(&seed_t, 255, 0, 999), q->c_last);
  } else {
    // by cust id
    q->by_last_name = FALSE;
    q->c_id = NURand(&seed_t, 1023, 0, TPCC_NCUST_PER_DIST - 1);
  }
#else
  q->by_last_name = FALSE;
  q->c_id = NURand(&seed_t, 1023, 0, TPCC_NCUST_PER_DIST - 1);
#endif
}

inline void TPCC::tpcc_get_next_delivery_query(int wid, void *arg) {
  static thread_local unsigned int seed_t = this->seed;
  struct tpcc_query *q = (struct tpcc_query *)arg;
  q->query_type = DELIVERY;
  q->w_id = wid;
  // q->d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);
  q->o_carrier_id = URand(&seed_t, 1, 10);
}

inline void TPCC::tpcc_get_next_stocklevel_query(int wid, void *arg) {
  static thread_local unsigned int seed_t = this->seed;
  struct tpcc_query *q = (struct tpcc_query *)arg;
  q->query_type = STOCK_LEVEL;
  q->w_id = wid;
  q->d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);
  q->threshold = URand(&seed_t, 10, 20);
}

bool TPCC::consistency_check_1() {
  // Check-1
  // Entries in the WAREHOUSE and DISTRICT tables must satisfy the relationship:
  // W_YTD = sum(D_YTD)

  // SQL: (EXPECTED 0 ROWS)
  //     SELECT sum(w_ytd), sum(d_ytd)
  //     FROM tpcc_warehouse, tpcc_district
  //     WHERE tpcc_warehouse.w_id = tpcc_district.d_w_id
  //     GROUP BY tpcc_warehouse.w_id
  //     HAVING sum(w_ytd) != sum(d_ytd)

  bool db_consistent = true;
  std::vector<proteus::thread> workers;
  // TODO: parallelize

  //  loaders.emplace_back(
  //      [this]() { this->create_tbl_warehouse(this->num_warehouse); });

  for (uint64_t i = 0; i < this->num_warehouse; i++) {
    // Get wh ytd.
    double wh_ytd = 0.0;
    const ushort w_col_scan[] = {8};  // position in columns
    auto w_idx_ptr = (global_conf::IndexVal *)table_warehouse->p_index->find(i);
    table_warehouse->getRecordByKey(w_idx_ptr->VID, w_col_scan, 1, &wh_ytd);

    double district_ytd = 0.0;
    for (uint j = 0; j < TPCC_NDIST_PER_WH; j++) {
      // Get all district ytd and sum it.
      double tmp_d_ytd = 0.0;
      const ushort d_col_scan[] = {9};  // position in columns
      auto d_idx_ptr = (global_conf::IndexVal *)table_district->p_index->find(
          MAKE_DIST_KEY(i, j));
      table_district->getRecordByKey(d_idx_ptr->VID, d_col_scan, 1, &tmp_d_ytd);
      district_ytd += tmp_d_ytd;
    }

    if ((size_t)wh_ytd != (size_t)district_ytd) {
      LOG(INFO) << "FAILED CONSISTENCY CHECK-1"
                << "\n\tWH: " << i << "\n\twh_ytd: " << wh_ytd << " | "
                << "district_ytd: " << district_ytd;
      db_consistent = false;
    }
  }

  for (auto &th : workers) {
    th.join();
  }
  return db_consistent;
}

std::vector<PreparedStatement> TPCC::consistency_check_2_query() {
  RelBuilderFactory ctx_one{"tpcc_consistency_check_2_1"};
  RelBuilderFactory ctx_two{"tpcc_consistency_check_2_2"};
  RelBuilderFactory ctx_three{"tpcc_consistency_check_2_3"};
  CatalogParser &catalog = CatalogParser::getInstance();
  PreparedStatement o_max =
      ctx_one.getBuilder()
          .scan<AeolusRemotePlugin>("tpcc_order<block-remote>",
                                    {"o_w_id", "o_d_id", "o_id"},
                                    CatalogParser::getInstance())
          .router(32, RoutingPolicy::LOCAL, DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_w_id"], arg["o_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          .pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_w_id"], arg["o_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["o_w_id"], arg["o_d_id"], arg["o_id"]};
              },
              {direction::ASC, direction::ASC, direction::NONE})
          .print([&](const auto &arg) -> std::vector<expression_t> {
            return {
                arg["o_w_id"],
                arg["o_d_id"],
                arg["o_id"],
            };
          })
          .prepare();

  PreparedStatement no_max =
      ctx_two.getBuilder()
          .scan<AeolusRemotePlugin>("tpcc_neworder<block-remote>",
                                    {"no_w_id", "no_d_id", "no_o_id"},
                                    CatalogParser::getInstance())
          .router(32, RoutingPolicy::LOCAL, DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["no_o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          .pack()
          .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .unpack()
          .groupby(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"]};
              },
              [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                return {
                    GpuAggrMatExpr{(arg["no_o_id"]), 1, 0, MAX},
                };
              },
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + 1,
              log2(this->num_warehouse * TPCC_NDIST_PER_WH) + (1024 * 1024))
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["no_w_id"], arg["no_d_id"], arg["no_o_id"]};
              },
              {direction::ASC, direction::ASC, direction::NONE})
          .print([&](const auto &arg) -> std::vector<expression_t> {
            return {
                arg["no_w_id"],
                arg["no_d_id"],
                arg["no_o_id"],
            };
          })
          .prepare();

  PreparedStatement d_next_oid =
      ctx_three.getBuilder()
          .scan<AeolusRemotePlugin>("tpcc_district<block-remote>",
                                    {"d_w_id", "d_id", "d_next_o_id"},
                                    CatalogParser::getInstance())
          .unpack()
          .sort(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg["d_w_id"], arg["d_id"], arg["d_next_o_id"]};
              },
              {direction::ASC, direction::ASC, direction::NONE})
          .project([&](const auto &arg) -> std::vector<expression_t> {
            return {(arg["d_w_id"]).as("PelagoProject#13144", "d_w_id"),
                    (arg["d_id"]).as("PelagoProject#13144", "d_id"),
                    (arg["d_next_o_id"] - ((int64_t)1))
                        .as("PelagoProject#13144", "d_next_o_id")};
          })
          .print([&](const auto &arg) -> std::vector<expression_t> {
            return {
                arg["d_w_id"],
                arg["d_id"],
                arg["d_next_o_id"],
            };
          })
          .prepare();

  return {o_max, no_max, d_next_oid};
}
static void print_inconsistency(std::string a, std::string b) {
  std::stringstream check1(a);
  std::string i1;

  std::stringstream check2(b);
  std::string i2;

  // Tokenizing w.r.t. space ' '
  while (getline(check1, i1, '\n') && getline(check2, i2, '\n')) {
    if (i1 != i2) {
      LOG(INFO) << i1 << " | " << i2;
    }
  }
}

bool TPCC::consistency_check_2() {
  // Check-2
  // Entries in the DISTRICT, ORDER, and NEW-ORDER tables must satisfy the
  // relationship: D_NEXT_O_ID - 1 = max(O_ID) = max(NO_O_ID)
  // for each district defined by
  // (D_W_ID = O_W_ID = NO_W_ID) and (D_ID = O_D_ID = NO_D_ID).
  // This condition does not apply to the NEW-ORDER table for any districts
  // which have no outstanding new orders (i.e., the number of rows is zero).

  //  Comment 2: For Consistency Conditions 2 and 4 (Clauses 3.3.2.2
  //  and 3.3.2.4), sampling the first, last, and two random warehouses is
  //  sufficient.
  bool db_consistent = true;

  auto queries = consistency_check_2_query();

  std::ostringstream stream_orders;
  std::ostringstream stream_new_orders;
  std::ostringstream stream_district_orders;

  stream_orders << queries[0].execute();
  stream_new_orders << queries[1].execute();
  stream_district_orders << queries[2].execute();

  std::string s_orders = stream_orders.str();
  std::string s_new_orders = stream_new_orders.str();
  std::string s_dist_orders = stream_district_orders.str();

  if (s_orders == s_dist_orders) {
    if (s_new_orders != s_dist_orders) {
      LOG(INFO) << "NewOrders and District Orders doesnt match.";
      db_consistent = false;
      LOG(INFO) << "NewOrders, DistOrders";
      print_inconsistency(s_new_orders, s_dist_orders);
    }
  } else {
    LOG(INFO) << "Orders and District Orders doesnt match.";
    db_consistent = false;
    LOG(INFO) << "Orders, DistOrders";
    print_inconsistency(s_orders, s_dist_orders);
  }
  return db_consistent;
}
bool TPCC::consistency_check_3() {
  // Check-3
  // Entries in the NEW-ORDER table must satisfy the relationship:
  // max(NO_O_ID) - min(NO_O_ID) + 1 =
  //                       [# of rows in the NEW-ORDER table for this district]
  // for each district defined by NO_W_ID and NO_D_ID. This condition does not
  // apply to any districts which have no outstanding new orders
  // (i.e., the number of rows is zero).

  // SQL: (EXPECTED 0 ROWS)
  //     SELECT no_w_id, no_d_id, max(no_o_id)-min(no_o_id)+1, count(no_o_id)
  //     FROM tpcc_new_order
  //     GROUP BY no_w_id, no_d_id
  //     HAVING max(no_o_id)-min(no_o_id)+1 != count(no_o_id)

  bool db_consistent = true;
  std::vector<proteus::thread> workers;

  // TODO: parallelize

  for (uint i = 0; i < this->num_warehouse; i++) {
    for (uint j = 0; j < TPCC_NDIST_PER_WH; j++) {
      size_t min_no_o_id = 0;
      size_t max_no_o_id = 0;
      size_t count_no_o_id = 0;
      // TODO: calculate.

      if ((max_no_o_id - min_no_o_id + 1) != count_no_o_id) {
        LOG(INFO) << "FAILED CONSISTENCY CHECK-3"
                  << "\n\tWH: " << i << " | DT: " << j
                  << "\n\t(max_no_o_id - min_no_o_id + 1): "
                  << (max_no_o_id - min_no_o_id + 1)
                  << " | count: " << count_no_o_id;
        db_consistent = false;
      }
    }
  }

  for (auto &th : workers) {
    th.join();
  }
  return db_consistent;
}
bool TPCC::consistency_check_4() {
  // Check-4
  // Entries in the ORDER and ORDER-LINE tables must satisfy the relationship:
  // sum(O_OL_CNT) = [number of rows in the ORDER-LINE table for this district]
  // for each district defined by (O_W_ID = OL_W_ID) and (O_D_ID = OL_D_ID).

  //  Comment 2: For Consistency Conditions 2 and 4 (Clauses 3.3.2.2
  //  and 3.3.2.4), sampling the first, last, and two random warehouses is
  //  sufficient.

  // SQL: (EXPECTED 0 ROWS)
  //     SELECT o_w_id, o_d_id, o_count, ol_count
  //     FROM (
  //	    SELECT o_w_id, o_d_id, sum(o_ol_cnt) as o_count
  //	    FROM tpcc_orders
  //	    GROUP BY o_w_id, o_d_id) o,
  //	    (SELECT ol_w_id, ol_d_id, count(o_ol_id) as ol_count
  //	    FROM tpcc_orderline
  //	    GROUP BY ol_w_id, ol_d_id) ol
  //     WHERE o.o_w_id = ol.ol_w_id
  //     AND o.o_d_id = ol.ol_d_id
  //     AND o_count != ol_count

  bool db_consistent = true;
  std::vector<proteus::thread> workers;

  // TODO: parallelize

  for (uint i = 0; i < this->num_warehouse; i++) {
    for (uint j = 0; j < TPCC_NDIST_PER_WH; j++) {
      size_t o_ol_sum = 0;
      // TODO: Get sum(o_ol_cnt)

      // TODO: Get count(ol_o_id)
      size_t ol_count = 0;

      if (o_ol_sum != ol_count) {
        LOG(INFO) << "FAILED CONSISTENCY CHECK-4"
                  << "\n\tWH: " << i << " | DT: " << j
                  << "\n\to_ol_count: " << o_ol_sum
                  << " | ol_count: " << ol_count;
        db_consistent = false;
      }
    }
  }

  for (auto &th : workers) {
    th.join();
  }
  return db_consistent;
}

void TPCC::verify_consistency() {
  LOG(INFO) << "Verifying consistency...";
  // NOTE: Only first-four are required, others are extra.

  //  Comment 1: The consistency conditions were chosen so that they would
  //  remain valid within the context of a larger order-entry application that
  //  includes the five TPC-C transactions (See Clause 1.1.). They are designed
  //  to be independent of the length of time for which such an application
  //  would be executed. Thus, for example, a condition involving I_PRICE was
  //  not included here since it is conceivable that within a larger application
  //  I_PRICE is modified from time to time.
  //
  //  Comment 2: For Consistency Conditions 2 and 4 (Clauses 3.3.2.2
  //  and 3.3.2.4), sampling the first, last, and two random warehouses is
  //  sufficient.

  // Set execution affinity to everywhere..
  //  cpu_set_t all_cpu_set;
  //  CPU_ZERO(&all_cpu_set);
  //  for (uint32_t i =0; i < topology::getInstance().getCoreCount(); i++)
  //    CPU_SET(i, &all_cpu_set);
  //  set_exec_location_on_scope d{all_cpu_set};

  // execute consistency checks.
  if (consistency_check_1() && consistency_check_2() /*&& consistency_check_3() &&
      consistency_check_4()*/) {
    LOG(INFO) << "DB IS CONSISTENT.";
  } else {
    LOG(FATAL) << "DB IS NOT CONSISTENT.";
  }
}
}  // namespace bench
