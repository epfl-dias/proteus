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

#include <chrono>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <olap/routing/degree-of-parallelism.hpp>
#include <oltp/common/utils.hpp>
#include <oltp/storage/layout/column_store.hpp>
#include <oltp/storage/storage-utils.hpp>
#include <string>

#include "tpcc/tpcc_64.hpp"

namespace bench {

bool TPCC::exec_txn(txn::Txn &txn, master_version_t master_ver,
                    delta_id_t delta_ver, partition_id_t partition_id) {
  //    const void *stmts, xid_t xid, master_version_t master_ver,
  //                    delta_id_t delta_ver, partition_id_t partition_id) {
  auto *q = (struct tpcc_query *)txn.stmts;
  switch (q->query_type) {
    case NEW_ORDER:
      return exec_neworder_txn(q, txn, master_ver, delta_ver, partition_id);
      //    case PAYMENT:
      //      return exec_payment_txn(q, xid, master_ver, delta_ver,
      //      partition_id);
      //    case ORDER_STATUS:
      //      return exec_orderstatus_txn(q, xid, master_ver, delta_ver,
      //      partition_id);
      //    case DELIVERY:
      //      return exec_delivery_txn(q, xid, master_ver, delta_ver,
      //      partition_id);
      //    case STOCK_LEVEL:
      //      return exec_stocklevel_txn(q, xid, master_ver, delta_ver,
      //      partition_id);
    default:
      assert(false);
      break;
  }
  return true;
}

void TPCC::gen_txn(worker_id_t wid, void *q, partition_id_t partition_id) {
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

bool TPCC::exec_neworder_txn(const struct tpcc_query *q, txn::Txn &txn,
                             master_version_t master_ver, delta_id_t delta_ver,
                             partition_id_t partition_id) {
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
  assert(idx_w_locks[num_locks] != nullptr);
  if (idx_w_locks[num_locks]->write_lck.try_lock()) {
#if !tpcc_dist_txns
    assert(storage::StorageUtils::get_pid(idx_w_locks[num_locks]->VID) ==
           partition_id);
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

    assert(idx_w_locks[num_locks] != nullptr);
    bool e_false_s = false;
    if (idx_w_locks[num_locks]->write_lck.try_lock()) {
#if !tpcc_dist_txns
      assert(storage::StorageUtils::get_pid(idx_w_locks[num_locks]->VID) ==
             partition_id);
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
  const column_id_t stock_col_rw[] = {
      2, TPCC_NDIST_PER_WH + 3, TPCC_NDIST_PER_WH + 4, TPCC_NDIST_PER_WH + 5};

  for (uint32_t ol_number = 0; ol_number < q->ol_cnt; ol_number++) {
    uint32_t ol_quantity = q->item[ol_number].ol_quantity;
    assert((ol_number + 1) < num_locks);

    global_conf::IndexVal *st_idx_ptr = idx_w_locks[ol_number + 1];
    assert(st_idx_ptr != nullptr);

    st_idx_ptr->latch.acquire();

    table_stock->getIndexedRecord(txn.txnTs, *st_idx_ptr, &st_rec, stock_col_rw,
                                  4);

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

    table_stock->updateRecord(txn.txnTs.txn_start_time, st_idx_ptr, &st_rec,
                              delta_ver, stock_col_rw, 4, master_ver);

    // early release of locks?
    st_idx_ptr->latch.release();
  }

  /*
   * EXEC SQL UPDATE district SET d _next_o_id = :d _next_o_id + 1
   * WHERE d _id = :d_id AN D d _w _id = :w _id ;
   */

  const column_id_t dist_col_scan[] = {8, 10};
  const column_id_t dist_col_upd[] = {10};

  struct __attribute__((packed)) dist_read {
    double d_tax;
    uint32_t d_next_o_id;
  };
  struct dist_read dist_no_read = {};

  global_conf::IndexVal *d_idx_ptr = idx_w_locks[0];
  // assert(storage::StorageUtils::get_pid(idx_w_locks[0]->VID) ==
  // partition_id); assert(storage::StorageUtils::get_pid(d_idx_ptr->VID) ==
  // partition_id);

  d_idx_ptr->latch.acquire();

  table_district->getIndexedRecord(txn.txnTs, *d_idx_ptr, &dist_no_read,
                                   dist_col_scan, 2);

  uint32_t d_next_o_id_upd = dist_no_read.d_next_o_id + 1;
  table_district->updateRecord(txn.txnTs.txn_start_time, d_idx_ptr,
                               &d_next_o_id_upd, delta_ver, dist_col_upd, 1,
                               master_ver);

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
  const column_id_t w_col_scan[] = {7};  // position in columns
  // std::cout << "WID::" << w_id << std::endl;
  global_conf::IndexVal *w_idx_ptr =
      (global_conf::IndexVal *)table_warehouse->p_index->find(
          (uint64_t)q->w_id);
#if !tpcc_dist_txns
  assert(storage::StorageUtils::get_pid(w_idx_ptr->VID) == partition_id);
#endif

  w_idx_ptr->latch.acquire();
  table_warehouse->getIndexedRecord(txn.txnTs, *w_idx_ptr, &w_tax, w_col_scan,
                                    1);
  w_idx_ptr->latch.release();

  // CUSTOMER

  // const ushort cust_col_scan[] = {5, 13, 15};
  const column_id_t cust_col_scan[] = {15};
  struct __attribute__((packed)) cust_read {
    // char c_last[LAST_NAME_LEN + 1];
    // char c_credit[2];
    double c_discount;
  };
  struct cust_read cust_no_read = {};

  global_conf::IndexVal *c_idx_ptr =
      (global_conf::IndexVal *)table_customer->p_index->find(
          (uint64_t)MAKE_CUST_KEY(q->w_id, q->d_id, q->c_id));
  assert(c_idx_ptr != nullptr);

  c_idx_ptr->latch.acquire();

  // if (storage::StorageUtils::get_pid(c_idx_ptr->VID) != partition_id) {
  //   std::unique_lock<std::mutex> lk(print_mutex);
  //   std::cout << "q->c_id: " << q->c_id << std::endl;
  //   std::cout << "q->w_id: " << q->w_id << std::endl;
  //   std::cout << "q->d_id: " << q->d_id << std::endl;

  //   std::cout << "partition_id: " << partition_id << std::endl;

  //   table_customer->reportUsage();
  // }

#if !tpcc_dist_txns
  assert(storage::StorageUtils::get_pid(c_idx_ptr->VID) == partition_id);
#endif

  table_customer->getIndexedRecord(txn.txnTs, *c_idx_ptr, &cust_no_read,
                                   cust_col_scan, 1);
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

  void *o_idx_ptr = table_order->insertRecord(&o_r, txn.txnTs.txn_start_time,
                                              partition_id, master_ver);

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

  void *no_idx_ptr = table_new_order->insertRecord(
      &no_r, txn.txnTs.txn_start_time, partition_id, master_ver);
#if index_on_order_tbl
  table_new_order->p_index->insert(order_key, no_idx_ptr);
#endif

  const column_id_t i_col_scan[] = {3};

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
    table_item->getIndexedRecord(txn.txnTs, *item_idx_ptr, &i_price, i_col_scan,
                                 1);
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

  auto *ol_idx_ptr_batch =
      (global_conf::IndexVal *)table_order_line->insertRecordBatch(
          ol_ptr, q->ol_cnt, TPCC_MAX_OL_PER_ORDER, txn.txnTs.txn_start_time,
          partition_id, master_ver);
#if index_on_order_tbl
  for (uint ol_number = 0; ol_number < q->ol_cnt; ol_number++) {
    void *pt = (void *)(ol_idx_ptr_batch + ol_number);
    table_order_line->p_index->insert(ol_key_batch[ol_number], pt);
  }
#endif
#endif

  return true;
}

// bool TPCC::exec_orderstatus_txn(const struct tpcc_query *q, xid_t xid,
//                                master_version_t master_ver,
//                                delta_id_t delta_ver,
//                                partition_id_t partition_id) {
//  // int ol_number = 1;
//
//  uint32_t cust_id = std::numeric_limits<uint32_t>::max();
//
//  if (q->by_last_name) {
//    /*
//      EXEC SQL SELECT count(c_id) INTO :namecnt
//            FROM customer
//            WHERE c_last=:c_last AND c_d_id=:d_id AND c_w_id=:w_id;
//
//      EXEC SQL DECLARE c_name CURSOR FOR
//          SELECT c_balance, c_first, c_middle, c_id
//          FROM customer
//          WHERE c_last=:c_last AND c_d_id=:d_id AND c_w_id=:w_id
//          ORDER BY c_first;
//      EXEC SQL OPEN c_name;
//
//      if (namecnt%2) namecnt++; / / Locate midpoint customer
//
//      for (n=0; n<namecnt/ 2; n++)
//      {
//        EXEC SQL FETCH c_name
//        INTO :c_balance, :c_first, :c_middle, :c_id;
//      }
//
//      EXEC SQL CLOSE c_name;
//     */
//
//    struct secondary_record sr {};
//
//    if (!this->cust_sec_index->find(
//            cust_derive_key(q->c_last, q->c_d_id, q->c_w_id), sr)) {
//      assert(false && "read_txn aborted! how!");
//      return false;
//    }
//
//    assert(sr.sr_idx < MAX_OPS_PER_QUERY);
//
//    struct cust_read c_recs[MAX_OPS_PER_QUERY];
//
//    uint nmatch = fetch_cust_records(sr, c_recs, xid, delta_ver);
//
//    assert(nmatch > 0);
//
//    // now sort based on first name and get middle element
//    // XXX: Inefficient bubble sort for now. Also the strcmp below has a huge
//    // overhead. We need some sorted secondary index structure
//    for (int i = 0; i < nmatch; i++) {
//      for (int j = i + 1; j < nmatch; j++) {
//        if (strcmp(c_recs[i].c_first, c_recs[j].c_first) > 0) {
//          struct cust_read tmp = c_recs[i];
//          c_recs[i] = c_recs[j];
//          c_recs[j] = tmp;
//        }
//      }
//    }
//
//    cust_id = MAKE_CUST_KEY(c_recs[nmatch / 2].c_w_id,
//                            c_recs[nmatch / 2].c_d_id, c_recs[nmatch /
//                            2].c_id);
//
//  } else {  // by cust_id
//
//    /*
//     * EXEC SQL SELECT c_balance, c_first, c_middle, c_last
//     * INTO :c_balance, :c_first, :c_middle, :c_last
//     * FROM customer
//     *  WHERE c_id=:c_id AND c_d_id=:d_id AND c_w_id=:w_id;
//     */
//
//    cust_id = MAKE_CUST_KEY(q->c_w_id, q->c_d_id, q->c_id);
//
//    assert(cust_id != std::numeric_limits<uint32_t>::max());
//
//    global_conf::IndexVal *c_idx_ptr =
//        (global_conf::IndexVal *)table_customer->p_index->find(cust_id);
//
//    assert(c_idx_ptr != nullptr);
//
//    struct __attribute__((packed)) cust_read_t {
//      char c_first[FIRST_NAME_LEN + 1];
//      char c_middle[2];
//      char c_last[LAST_NAME_LEN + 1];
//      double c_balance;
//    };
//
//    const column_id_t cust_col_scan_read[] = {3, 4, 5, 15};
//    struct cust_read_t c_r {};
//
//    c_idx_ptr->latch.acquire();
//
//    table_customer->getIndexedRecord(txn.txnTs, c_idx_ptr, &c_r,
//    cust_col_scan_read,
//                                     4);
//    c_idx_ptr->latch.release();
//
//    /* EXEC SQL SELECT o_id, o_carrier_id, o_entry_d
//     * INTO :o_id, :o_carrier_id, :entdate
//     * FROM orders
//     *  ORDER BY o_id DESC;
//     */
//
//    /*
//     * EXEC SQL DECLARE c_line CURSOR FOR
//     * SELECT ol_i_id, ol_supply_w_id, ol_quantity,
//         ol_amount, ol_delivery_d
//     * FROM order_line
//     *  WHERE ol_o_id=:o_id AND ol_d_id=:d_id AND ol_w_id=:w_id;
//     */
//
//    int counter1 = 0;
//
//    struct __attribute__((packed)) p_order_read {
//      uint32_t o_id;
//      uint32_t o_carrier_id;
//      uint32_t entdate;
//      uint32_t o_ol_cnt;
//    } p_or[TPCC_NCUST_PER_DIST]{};
//
//    struct __attribute__((packed)) p_orderline_read {
//      uint32_t ol_o_id;
//      uint32_t ol_supply_w_id;
//      date_t ol_delivery_d;
//      uint32_t ol_quantity;
//      double ol_amount;
//    } p_ol_r{};
//
//    const column_id_t o_col_scan[] = {0, 4, 5, 6};
//    const column_id_t ol_col_scan[] = {0, 5, 6, 7, 8};
//
//    for (int o = TPCC_NCUST_PER_DIST - 1, i = 0; o > 0; o--, i++) {
//      global_conf::IndexVal *o_idx_ptr =
//          (global_conf::IndexVal *)table_order->p_index->find(
//              MAKE_ORDER_KEY(q->w_id, q->d_id, o));
//
//      assert(o_idx_ptr != nullptr);
//
//      o_idx_ptr->latch.acquire();
//      table_order->getIndexedRecord(txn.txnTs, o_idx_ptr, &p_or[i],
//      o_col_scan, 4); o_idx_ptr->latch.release();
//
//      for (auto ol_number = 0; ol_number < p_or[i].o_ol_cnt; ++ol_number) {
//        global_conf::IndexVal *ol_idx_ptr =
//            (global_conf::IndexVal *)table_order_line->p_index->find(
//                MAKE_OL_KEY(q->w_id, q->d_id, p_or[i].o_id, ol_number));
//
//        assert(ol_idx_ptr != nullptr);
//
//        ol_idx_ptr->latch.acquire();
//        table_order_line->getIndexedRecord(txn.txnTs, ol_idx_ptr, &p_ol_r,
//                                           ol_col_scan, 5);
//        ol_idx_ptr->latch.release();
//      }
//    }
//  }
//
//  return true;
//}
//
// bool TPCC::exec_stocklevel_txn(const struct tpcc_query *q, xid_t xid,
//                               master_version_t master_ver,
//                               delta_id_t delta_ver,
//                               partition_id_t partition_id) {
//  /*
//  This transaction accesses
//  order_line,
//  stock,
//  district
//  */
//
//  /*
//   * EXEC SQL SELECT d_next_o_id INTO :o_id
//   * FROM district
//   * WHERE d_w_id=:w_id AND d_id=:d_id;
//   */
//
//  global_conf::IndexVal *d_idx_ptr =
//      (global_conf::IndexVal *)table_district->p_index->find(
//          MAKE_DIST_KEY(q->w_id, q->d_id));
//
//  assert(d_idx_ptr != nullptr);
//
//  const column_id_t dist_col_scan_read[] = {10};
//  uint32_t o_id = 0;
//
//  d_idx_ptr->latch.acquire();
//  table_district->getIndexedRecord(txn.txnTs, d_idx_ptr, &o_id,
//  dist_col_scan_read,
//                                   1);
//  d_idx_ptr->latch.release();
//
//  /*
//   * EXEC SQL SELECT COUNT(DISTINCT (s_i_id)) INTO :stock_count
//   * FROM order_line, stock
//   * WHERE ol_w_id=:w_id AND
//   * ol_d_id=:d_id AND ol_o_id<:o_id AND
//   * ol_o_id>=:o_id-20 AND s_w_id=:w_id AND
//   * s_i_id=ol_i_id AND s_quantity < :threshold;
//   */
//
//  uint32_t ol_o_id = o_id - 20;
//  uint32_t ol_number = -1;
//  uint32_t stock_count = 0;
//
//  const column_id_t ol_col_scan_read[] = {4};
//  const column_id_t st_col_scan_read[] = {2};
//
//  while (ol_o_id < o_id) {
//    while (ol_number < TPCC_MAX_OL_PER_ORDER) {
//      ol_number++;
//
//      // orderline first
//      global_conf::IndexVal *ol_idx_ptr =
//          (global_conf::IndexVal *)table_order_line->p_index->find(
//              MAKE_OL_KEY(q->w_id, q->d_id, ol_o_id, ol_number));
//
//      if (ol_idx_ptr == nullptr) continue;
//
//      uint32_t ol_i_id;
//      int32_t s_quantity;
//
//      ol_idx_ptr->latch.acquire();
//      table_order_line->getIndexedRecord(txn.txnTs, ol_idx_ptr, &ol_i_id,
//                                         ol_col_scan_read, 1);
//      ol_idx_ptr->latch.release();
//
//      // stock
//      global_conf::IndexVal *st_idx_ptr =
//          (global_conf::IndexVal *)table_stock->p_index->find(
//              MAKE_STOCK_KEY(q->w_id, ol_i_id));
//
//      assert(st_idx_ptr != nullptr);
//
//      st_idx_ptr->latch.acquire();
//      table_stock->getIndexedRecord(txn.txnTs, st_idx_ptr, &s_quantity,
//                                    st_col_scan_read, 1);
//      st_idx_ptr->latch.release();
//
//      if (s_quantity < q->threshold) {
//        stock_count++;
//      }
//    }
//    ol_o_id = ol_o_id + 1;
//  }
//
//  return true;
//}
//
// inline uint TPCC::fetch_cust_records(const struct secondary_record &sr,
//                                     struct cust_read *c_recs, xid_t xid,
//                                     delta_id_t delta_ver) {
//  uint nmatch = 0;
//
//  // struct cust_read {
//  //   uint32_t c_id;
//  //   ushort c_d_id;
//  //   ushort c_w_id;
//  //   char c_first[FIRST_NAME_LEN + 1];
//  //   char c_last[LAST_NAME_LEN + 1];
//  // };
//
//  const column_id_t c_col_scan[] = {0, 1, 2, 3, 5};  // position in columns
//
//  for (int i = 0; i < sr.sr_nids; i++) {
//    global_conf::IndexVal *c_idx_ptr =
//        (global_conf::IndexVal *)table_customer->p_index->find(
//            (uint64_t)(sr.sr_rids[i]));
//
//    assert(c_idx_ptr != nullptr);
//
//    c_idx_ptr->latch.acquire();
//    table_customer->getIndexedRecord(txn.txnTs, c_idx_ptr, &c_recs[nmatch],
//                                     c_col_scan, 5);
//    c_idx_ptr->latch.release();
//
//    nmatch++;
//  }
//
//  return nmatch;
//}
//
// bool TPCC::exec_payment_txn(const struct tpcc_query *q, xid_t xid,
//                            master_version_t master_ver, delta_id_t delta_ver,
//                            partition_id_t partition_id) {
//  // // Updates..
//
//  // // update warehouse (ytd) / district (ytd),
//  // // update customer (balance/c_data)
//  // // insert history
//
//  // /*====================================================+
//  //     Acquire Locks for Warehouse, District, Customer
//  // +====================================================*/
//
//  // Lock Warehouse
//  global_conf::IndexVal *w_idx_ptr =
//      (global_conf::IndexVal *)table_warehouse->p_index->find(q->w_id);
//
//  assert(w_idx_ptr != nullptr);
//  bool e_false = false;
//  if (!(w_idx_ptr->write_lck.try_lock())) {
//    return false;
//  }
//
//  // Lock District
//  global_conf::IndexVal *d_idx_ptr =
//      (global_conf::IndexVal *)table_district->p_index->find(
//          MAKE_DIST_KEY(q->w_id, q->d_id));
//
//  assert(d_idx_ptr != nullptr);
//  e_false = false;
//  if (!(d_idx_ptr->write_lck.try_lock())) {
//    w_idx_ptr->write_lck.unlock();
//    return false;
//  }
//
//  // Lock Customer
//
//  // some logic is required here as customer can be by name or by id..
//
//  uint32_t cust_id = std::numeric_limits<uint32_t>::max();
//
//  if (q->by_last_name) {
//    /*==========================================================+
//      EXEC SQL SELECT count(c_id) INTO :namecnt
//      FROM customer
//      WHERE c_last=:c_last AND c_d_id=:c_d_id AND c_w_id=:c_w_id;
//      +==========================================================*/
//
//    struct secondary_record sr {};
//
//    if (!this->cust_sec_index->find(
//            cust_derive_key(q->c_last, q->c_d_id, q->c_w_id), sr)) {
//      // ABORT
//
//      w_idx_ptr->write_lck.unlock();
//      d_idx_ptr->write_lck.unlock();
//
//      return false;
//    }
//
//    assert(sr.sr_idx < MAX_OPS_PER_QUERY);
//
//    struct cust_read c_recs[MAX_OPS_PER_QUERY];
//
//    uint nmatch = fetch_cust_records(sr, c_recs, xid, delta_ver);
//
//    assert(nmatch > 0);
//
//    /*============================================================================+
//        for (n=0; n<namecnt/2; n++) {
//            EXEC SQL FETCH c_byname
//            INTO :c_first, :c_middle, :c_id,
//                 :c_street_1, :c_street_2, :c_city, :c_state, :c_zip,
//                 :c_phone, :c_credit, :c_credit_lim, :c_discount, :c_balance,
//    :c_since;
//            }
//        EXEC SQL CLOSE c_byname;
//    +=============================================================================*/
//    // now sort based on first name and get middle element
//    // XXX: Inefficient bubble sort for now. Also the strcmp below has a huge
//    // overhead. We need some sorted secondary index structure
//    for (int i = 0; i < nmatch; i++) {
//      for (int j = i + 1; j < nmatch; j++) {
//        if (strcmp(c_recs[i].c_first, c_recs[j].c_first) > 0) {
//          struct cust_read tmp = c_recs[i];
//          c_recs[i] = c_recs[j];
//          c_recs[j] = tmp;
//        }
//      }
//    }
//
//    cust_id = MAKE_CUST_KEY(c_recs[nmatch / 2].c_w_id,
//                            c_recs[nmatch / 2].c_d_id, c_recs[nmatch /
//                            2].c_id);
//
//  } else {  // by cust_id
//    cust_id = MAKE_CUST_KEY(q->c_w_id, q->c_d_id, q->c_id);
//  }
//
//  assert(cust_id != std::numeric_limits<uint32_t>::max());
//
//  global_conf::IndexVal *c_idx_ptr =
//      (global_conf::IndexVal *)table_customer->p_index->find(cust_id);
//
//  assert(c_idx_ptr != nullptr);
//  e_false = false;
//  if (!(c_idx_ptr->write_lck.try_lock())) {
//    w_idx_ptr->write_lck.unlock();
//    d_idx_ptr->write_lck.unlock();
//    return false;
//  }
//
//  // -------------  ALL LOCKS ACQUIRED
//
//  /*====================================================+
//      EXEC SQL UPDATE warehouse SET w_ytd = w_ytd + :h_amount
//      WHERE w_id=:w_id;
//  +====================================================*/
//  /*===================================================================+
//      EXEC SQL SELECT w_street_1, w_street_2, w_city, w_state, w_zip, w_name
//      INTO :w_street_1, :w_street_2, :w_city, :w_state, :w_zip, :w_name
//      FROM warehouse
//      WHERE w_id=:w_id;
//  +===================================================================*/
//
//  // update warehouse and then release the write lock
//
//  double w_ytd = 0.0;
//  const column_id_t wh_col_scan_upd[] = {7};
//  w_idx_ptr->latch.acquire();
//  table_warehouse->getIndexedRecord(txn.txnTs, w_idx_ptr, &w_ytd,
//  wh_col_scan_upd, 1);
//
//  w_ytd += q->h_amount;
//  table_warehouse->updateRecord(xid, w_idx_ptr, &w_ytd, delta_ver,
//                                wh_col_scan_upd, 1, master_ver);
//  w_idx_ptr->write_lck.unlock();
//  w_idx_ptr->latch.release();
//
//  /*=====================================================+
//      EXEC SQL UPDATE district SET d_ytd = d_ytd + :h_amount
//      WHERE d_w_id=:w_id AND d_id=:d_id;
//  =====================================================*/
//  /*====================================================================+
//      EXEC SQL SELECT d_street_1, d_street_2, d_city, d_state, d_zip, d_name
//      INTO :d_street_1, :d_street_2, :d_city, :d_state, :d_zip, :d_name
//      FROM district
//      WHERE d_w_id=:w_id AND d_id=:d_id;
//  +====================================================================*/
//
//  double d_ytd = 0.0;
//  const column_id_t d_col_scan_upd[] = {8};
//  d_idx_ptr->latch.acquire();
//
//  table_district->getIndexedRecord(txn.txnTs, d_idx_ptr, &d_ytd,
//  d_col_scan_upd, 1); d_ytd += q->h_amount;
//
//  table_district->updateRecord(xid, d_idx_ptr, &d_ytd, delta_ver,
//                               d_col_scan_upd, 1, master_ver);
//
//  d_idx_ptr->write_lck.unlock();
//  d_idx_ptr->latch.release();
//
//  //---
//
//  /*======================================================================+
//    EXEC SQL UPDATE customer SET c_balance = :c_balance, c_data = :c_new_data
//    WHERE c_w_id = :c_w_id AND c_d_id = :c_d_id AND c_id = :c_id;
//    +======================================================================*/
//
//  struct __attribute__((packed)) cust_rw_t {
//    double c_balance;
//    double c_ytd_payment;
//    uint32_t c_payment_cnt;
//    char c_data[501];
//    char c_credit[2];
//  };
//
//  const column_id_t cust_col_scan_read[] = {15, 16, 17, 19, 12};
//  ushort num_col_upd = 3;
//
//  struct cust_rw_t cust_rw = {};
//
//  c_idx_ptr->latch.acquire();
//  table_customer->getIndexedRecord(txn.txnTs, c_idx_ptr, &cust_rw,
//  cust_col_scan_read,
//                                   5);
//
//  if (cust_rw.c_credit[0] == 'B' && cust_rw.c_credit[1] == 'C') {
//    num_col_upd++;
//
//    /*=====================================================+
//      EXEC SQL SELECT c_data
//            INTO :c_data
//            FROM customer
//            WHERE c_w_id=:c_w_id AND c_d_id=:c_d_id AND c_id=:c_id;
//            +=====================================================*/
//    // char c_new_data[501];
//    sprintf(cust_rw.c_data, "| %4d %2d %4d %2d %4d $%7.2f", q->c_id,
//    q->c_d_id,
//            q->c_w_id, q->d_id, q->w_id, q->h_amount);
//    strncat(cust_rw.c_data, cust_rw.c_data, 500 - strlen(cust_rw.c_data));
//  }
//
//  table_customer->updateRecord(xid, c_idx_ptr, &cust_rw, delta_ver,
//                               cust_col_scan_read, num_col_upd, master_ver);
//
//  c_idx_ptr->write_lck.unlock();
//  c_idx_ptr->latch.release();
//
//  /*
//      char h_data[25];
//      char * w_name = r_wh_local->get_value("W_NAME");
//      char * d_name = r_dist_local->get_value("D_NAME");
//      strncpy(h_data, w_name, 10);
//      int length = strlen(h_data);
//      if (length > 10) length = 10;
//      strcpy(&h_data[length], "    ");
//      strncpy(&h_data[length + 4], d_name, 10);
//      h_data[length+14] = '\0';
//  */
//
//  /*=============================================================================+
//    EXEC SQL INSERT INTO
//    history (h_c_d_id, h_c_w_id, h_c_id, h_d_id, h_w_id, h_date, h_amount,
//    h_data) VALUES (:c_d_id, :c_w_id, :c_id, :d_id, :w_id, :datetime,
//    :h_amount, :h_data);
//    +=============================================================================*/
//
//  struct tpcc_history h_ins {};
//  h_ins.h_c_id = q->c_id;
//  h_ins.h_c_d_id = q->c_d_id;
//  h_ins.h_c_w_id = q->c_w_id;
//  h_ins.h_d_id = q->d_id;
//  h_ins.h_w_id = q->w_id;
//  h_ins.h_date = get_timestamp();
//  h_ins.h_amount = q->h_amount;
//
//  void *hist_idx_ptr =
//      table_history->insertRecord(&h_ins, xid, partition_id, master_ver);
//
//  return true;
//}

bool TPCC::exec_delivery_txn(const struct tpcc_query *q, txn::Txn &txn,
                             master_version_t master_ver, delta_id_t delta_ver,
                             partition_id_t partition_id) {
  static thread_local uint64_t delivered_orders[TPCC_NDIST_PER_WH] = {0};

  date_t delivery_d = get_timestamp();
  // FIXME: Broken
  //
  //  for (int d_id = 0; d_id < TPCC_NDIST_PER_WH; d_id++) {
  //    const uint64_t order_end_idx =
  //        MAKE_ORDER_KEY(q->w_id, d_id, TPCC_MAX_ORD_PER_DIST);
  //
  //    if (delivered_orders[d_id] == 0) {
  //      delivered_orders[d_id] = MAKE_ORDER_KEY(q->w_id, d_id, 0);
  //    }
  //
  //    for (size_t o = delivered_orders[d_id]; o <= order_end_idx; o++) {
  //      global_conf::IndexVal *idx_w_locks[TPCC_MAX_OL_PER_ORDER + 2] = {};
  //      uint num_locks = 0;
  //
  //      uint64_t okey = MAKE_ORDER_KEY(q->w_id, d_id, 0);
  //
  //      // get_new_order
  //      struct tpcc_new_order no_read = {};
  //
  //      global_conf::IndexVal *no_idx_ptr =
  //          (global_conf::IndexVal *)table_new_order->p_index->find(okey);
  //      if (no_idx_ptr == nullptr) {
  //        // LOG(INFO) << "W_ID: " << q->w_id << " | D_ID: " << d_id
  //        //           << " | order: " << o << "  .. DONE-S";
  //        break;
  //      }
  //
  //      no_idx_ptr->latch.acquire();
  //
  //      if (txn::CC_MV2PL::is_readable(no_idx_ptr->t_min, xid)) {
  //        table_new_order->getRecordByKey(no_idx_ptr->VID, nullptr, 0,
  //        &no_read);
  //
  //      } else {
  //        // Actually this is the breaking condition of loop.
  //        std::cout << "DONE." << std::endl;
  //        LOG(INFO) << "W_ID: " << q->w_id << " | D_ID: " << d_id
  //                  << " | order: " << o << "  .. DONE";
  //
  //        break;
  //      }
  //      no_idx_ptr->latch.release();
  //
  //      // Get Order
  //
  //      struct __attribute__((packed)) order_read_st {
  //        uint32_t o_c_id;
  //        uint32_t o_ol_cnt;
  //      };
  //      struct order_read_st order_read = {};
  //
  //      order_read.o_c_id = std::numeric_limits<uint32_t>::max();
  //
  //      const ushort order_col_scan[] = {3, 6};
  //
  //      global_conf::IndexVal *o_idx_ptr =
  //          (global_conf::IndexVal *)table_order->p_index->find(okey);
  //      assert(o_idx_ptr != NULL || o_idx_ptr != nullptr);
  //
  //      o_idx_ptr->latch.acquire();
  //
  //      if (txn::CC_MV2PL::is_readable(o_idx_ptr->t_min, xid)) {
  //        table_order->getRecordByKey(o_idx_ptr->VID, order_col_scan, 2,
  //                                    &order_read);
  //
  //      } else {
  //        assert(false && "impossible");
  //      }
  //      o_idx_ptr->latch.release();
  //
  //      assert(order_read.o_c_id != std::numeric_limits<uint32_t>::max());
  //      assert(order_read.o_ol_cnt != 0);
  //
  //      // lock on order, orderline and customer
  //
  //      // ACQUIRE WRITE_LOCK FOR CUSTOMER
  //      idx_w_locks[num_locks] =
  //          (global_conf::IndexVal *)table_customer->p_index->find(
  //              (uint64_t)MAKE_CUST_KEY(q->w_id, d_id, order_read.o_c_id));
  //
  //      bool e_false = false;
  //      assert(idx_w_locks[num_locks] != NULL ||
  //             idx_w_locks[num_locks] != nullptr);
  //      if (idx_w_locks[num_locks]->write_lck.try_lock()) {
  //#if !tpcc_dist_txns
  //        assert(storage::StorageUtils::get_pid(idx_w_locks[num_locks]->VID)
  //        == partition_id);
  //#endif
  //        num_locks++;
  //      } else {
  //        assert(false && "Not possible");
  //        return false;
  //      }
  //
  //      // ACQUIRE WRITE_LOCK FOR ORDER
  //      idx_w_locks[num_locks] =
  //          (global_conf::IndexVal *)table_order->p_index->find(okey);
  //
  //      e_false = false;
  //      assert(idx_w_locks[num_locks] != NULL ||
  //             idx_w_locks[num_locks] != nullptr);
  //      if (idx_w_locks[num_locks]->write_lck.try_lock()) {
  //#if !tpcc_dist_txns
  //        assert(storage::StorageUtils::get_pid(idx_w_locks[num_locks]->VID)
  //        == partition_id);
  //#endif
  //        num_locks++;
  //      } else {
  //        assert(false && "Not possible");
  //        return false;
  //      }
  //
  //      // ACQUIRE WRITE_LOCK FOR ORDER-LINE
  //      for (uint ol_number = 0; ol_number < order_read.o_ol_cnt; ol_number++)
  //      {
  //        idx_w_locks[num_locks] =
  //            (global_conf::IndexVal *)table_order_line->p_index->find(
  //                (uint64_t)MAKE_OL_KEY(q->w_id, d_id, okey, ol_number));
  //
  //        assert(idx_w_locks[num_locks] != NULL ||
  //               idx_w_locks[num_locks] != nullptr);
  //        bool e_false_s = false;
  //        if (idx_w_locks[num_locks]->write_lck.try_lock()) {
  //#if !tpcc_dist_txns
  //          assert(storage::StorageUtils::get_pid(idx_w_locks[num_locks]->VID)
  //          == partition_id);
  //#endif
  //          num_locks++;
  //        } else {
  //          txn::CC_MV2PL::release_locks(idx_w_locks, num_locks);
  //          assert(false && "Not possible");
  //          return false;
  //        }
  //      }
  //
  //      // update order
  //      constexpr ushort o_col_upd[] = {5};
  //      o_idx_ptr->latch.acquire();
  //      table_order->updateRecord(xid, o_idx_ptr, &q->o_carrier_id,
  //      master_ver,
  //                                delta_ver, o_col_upd, 1);
  //      o_idx_ptr->latch.release();
  //
  //      // update orderline
  //      //    upd delivery_d and read ol_amount
  //
  //      constexpr ushort ol_col_r[] = {8};
  //      constexpr ushort ol_col_upd[] = {6};
  //
  //      double ol_amount_sum = 0;
  //
  //      for (uint ol_number = 0; ol_number < order_read.o_ol_cnt; ol_number++)
  //      {
  //        global_conf::IndexVal *ol_idx_ptr = idx_w_locks[ol_number + 2];
  //        assert(ol_idx_ptr != NULL || ol_idx_ptr != nullptr);
  //
  //        ol_idx_ptr->latch.acquire();
  //
  //        if (txn::CC_MV2PL::is_readable(ol_idx_ptr->t_min, xid)) {
  //          double ol_amount = 0;
  //          table_order_line->getRecordByKey(ol_idx_ptr->VID, ol_col_r, 1,
  //                                           &ol_amount);
  //          ol_amount_sum += ol_amount;
  //        } else {
  //          assert(false && "Not possible");
  //        }
  //
  //        table_order_line->updateRecord(xid, ol_idx_ptr, &delivery_d,
  //        master_ver,
  //                                       delta_ver, ol_col_upd, 1);
  //        ol_idx_ptr->latch.release();
  //      }
  //
  //      // update customer
  //
  //      constexpr ushort c_col_rw[] = {15};
  //      double c_balance = 0;
  //      global_conf::IndexVal *c_idx_ptr = idx_w_locks[0];
  //      assert(c_idx_ptr != NULL || c_idx_ptr != nullptr);
  //
  //      c_idx_ptr->latch.acquire();
  //
  //      if (txn::CC_MV2PL::is_readable(c_idx_ptr->t_min, xid)) {
  //        double ol_amount = 0;
  //        table_customer->getRecordByKey(c_idx_ptr->VID, c_col_rw, 1,
  //        &c_balance); c_balance += ol_amount_sum;
  //      } else {
  //        assert(false && "Not possible");
  //      }
  //
  //      table_customer->updateRecord(xid, c_idx_ptr, &c_balance, master_ver,
  //                                   delta_ver, c_col_rw, 1);
  //      c_idx_ptr->latch.release();
  //
  //      // finally
  //      txn::CC_MV2PL::release_locks(idx_w_locks, num_locks);
  //      delivered_orders[d_id] = o;
  //    }
  //  }
  return true;
}

inline void TPCC::tpcc_get_next_neworder_query(int wid, void *arg) const {
  // mprotect(arg, sizeof(struct tpcc_query), PROT_WRITE);
  static thread_local unsigned int seed_t = this->seed;
  static thread_local unsigned int n_wh = this->num_warehouse;
  // static thread_local unsigned int n_actv_wh = this->num_active_workers;

#if PARTITION_LOCAL_ITEM_TABLE
  static thread_local auto start = wid * (TPCC_MAX_ITEMS / n_wh);
  static thread_local auto end = start + (TPCC_MAX_ITEMS / n_wh);
#endif

  int ol_cnt, dup;
  auto *q = (struct tpcc_query *)arg;

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

inline void TPCC::tpcc_get_next_orderstatus_query(int wid, void *arg) const {
  static thread_local unsigned int seed_t = this->seed;
  auto *q = (struct tpcc_query *)arg;
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

inline void TPCC::tpcc_get_next_payment_query(int wid, void *arg) const {
  static thread_local unsigned int seed_t = this->seed;
  auto *q = (struct tpcc_query *)arg;
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

inline void TPCC::tpcc_get_next_delivery_query(int wid, void *arg) const {
  static thread_local unsigned int seed_t = this->seed;
  auto *q = (struct tpcc_query *)arg;
  q->query_type = DELIVERY;
  q->w_id = wid;
  // q->d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);
  q->o_carrier_id = URand(&seed_t, 1, 10);
}

inline void TPCC::tpcc_get_next_stocklevel_query(int wid, void *arg) const {
  static thread_local unsigned int seed_t = this->seed;
  auto *q = (struct tpcc_query *)arg;
  q->query_type = STOCK_LEVEL;
  q->w_id = wid;
  q->d_id = URand(&seed_t, 0, TPCC_NDIST_PER_WH - 1);
  q->threshold = URand(&seed_t, 10, 20);
}

}  // namespace bench
