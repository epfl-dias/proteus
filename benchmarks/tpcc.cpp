#include "benchmarks/tpcc.hpp"
#include "benchmarks/bench_utils.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <functional>
#include <iostream>
#include <locale>
#include <string>
#include <thread>

namespace bench {

bool TPCC::exec_txn(void *stmts, uint64_t xid, ushort master_ver,
                    ushort delta_ver) {
  struct tpcc_query *q = (struct tpcc_query *)stmts;
  switch (q->query_type) {
    case NEW_ORDER:
      return exec_neworder_txn(q, xid, master_ver, delta_ver);
      break;
    case PAYMENT:
      return exec_payment_txn(q, xid, master_ver, delta_ver);
      break;
    case ORDER_STATUS:
      return exec_orderstatus_txn(q, xid, master_ver, delta_ver);
      break;
    case DELIVERY:
      return exec_delivery_txn(q, xid, master_ver, delta_ver);
      break;
    case STOCK_LEVEL:
      return exec_stocklevel_txn(q, xid, master_ver, delta_ver);
    default:
      return false;
      break;
  }
  return true;
}

void TPCC::gen_txn(int wid, void *q) {
  tpcc_get_next_neworder_query(wid, q);
  return;
  static thread_local uint sequence_counter = 0;

  switch (sequence[sequence_counter++ % MIX_COUNT]) {
    case NEW_ORDER:
      tpcc_get_next_neworder_query(wid, q);
      return;
    case PAYMENT:
      tpcc_get_next_payment_query(wid, q);
      return;
    case ORDER_STATUS:
      tpcc_get_next_orderstatus_query(wid, q);
      return;
    case DELIVERY:
      tpcc_get_next_delivery_query(wid, q);
      return;
    case STOCK_LEVEL:
      tpcc_get_next_stocklevel_query(wid, q);
    default:
      return;
  }
}

bool TPCC::exec_neworder_txn(struct tpcc_query *q, uint64_t xid,
                             ushort master_ver, ushort delta_ver) {
  char remote = q->remote;
  uint w_id = q->w_id;
  int d_id = q->d_id;
  int c_id = q->c_id;
  int ol_cnt = q->ol_cnt;
  int o_entry_d = q->o_entry_d;

  // print_tpcc_query(q);

  std::vector<global_conf::IndexVal *>
      hash_ptrs_lock_acquired;  // reserve space

  // TODO: Maybe create a fetch and update kind of functionality here
  // Move this as a first statement so that abort --> FAIL FAST
  /*
   * EXEC SQL SELECT d_next_o_id, d_tax
   *  INTO :d_next_o_id, :d_tax
   *  FROM district WHERE d_id = :d_id AND d_w_id = :w_id;
   */

  ushort dist_key = MAKE_DIST_KEY(w_id, d_id);

  // ACQUIRE WRITE_LOCK FOR DISTRICT

  global_conf::IndexVal *d_idx_ptr =
      (global_conf::IndexVal *)table_district->p_index->find(dist_key);

  bool e_false = false;
  assert(d_idx_ptr != NULL || d_idx_ptr != nullptr);
  if (d_idx_ptr->write_lck.compare_exchange_strong(e_false, true)) {
    hash_ptrs_lock_acquired.emplace_back(d_idx_ptr);
  } else {
    // ABORT
    // No need to free locks as until this point, this was the first
    // attempt to acquire a write lock
    // txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired);
    // std::cout << "Abort-1-" << w_id << std::endl;
    return false;
  }

  // ACQUIRE LOCK FOR STOCK

  for (int ol_number = 0; ol_number < ol_cnt; ol_number++) {
    uint64_t ol_i_id = q->item[ol_number].ol_i_id;
    uint64_t ol_supply_w_id = q->item[ol_number].ol_supply_w_id;

    uint32_t stock_key = MAKE_STOCK_KEY(ol_supply_w_id, ol_i_id);
    // ACQUIRE WRITE_LOCK FOR DISTRICT
    global_conf::IndexVal *st_idx_ptr =
        (global_conf::IndexVal *)table_stock->p_index->find(stock_key);

    assert(st_idx_ptr != NULL || st_idx_ptr != nullptr);
    bool e_false_s = false;
    if (st_idx_ptr->write_lck.compare_exchange_strong(e_false_s, true)) {
      hash_ptrs_lock_acquired.emplace_back(st_idx_ptr);
    } else {
      // ABORT
      txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired);
      return false;
    }
  }
  // Until this point, we have acquire all the necessary write locks,
  // we may now begin with reads and inserts which are never gonna abort in
  // MV2PL.

  for (int ol_number = 0; ol_number < ol_cnt; ol_number++) {
    int ol_i_id = q->item[ol_number].ol_i_id;
    int ol_supply_w_id = q->item[ol_number].ol_supply_w_id;
    int ol_quantity = q->item[ol_number].ol_quantity;

    // IDK why to read all the data when not using it in the
    // transaction. I am gonna skip it.
    // Read only useful shit: s_quantity, s_ytd, s_order_cnt, s_remote_cnt,

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
    uint32_t stock_key = MAKE_STOCK_KEY(ol_supply_w_id, ol_i_id);

    std::vector<int> st_col = {2, TPCC_NDIST_PER_WH + 3, TPCC_NDIST_PER_WH + 4,
                               TPCC_NDIST_PER_WH + 5};
    struct st_read {
      short s_quantity;
      ushort s_ytd;
      ushort s_order_cnt;
      ushort s_remote_cnt;
    };
    struct st_read st_rec;

    global_conf::IndexVal *st_idx_ptr =
        (global_conf::IndexVal *)table_stock->p_index->find(stock_key);
    assert(st_idx_ptr != NULL || st_idx_ptr != nullptr);
    st_idx_ptr->latch.acquire();

    if (txn::CC_MV2PL::is_readable(st_idx_ptr->t_min, st_idx_ptr->t_max, xid)) {
      table_stock->getRecordByKey(st_idx_ptr->VID, st_idx_ptr->last_master_ver,
                                  &st_col, &st_rec);
    } else {
      // std::cout << "not readable 1" << std::endl;
      // std::cout << "t_min: " << st_idx_ptr->t_min << std::endl;
      // std::cout << "t_max: " << st_idx_ptr->t_max << std::endl;
      // std::cout << "xid: " << xid << std::endl;

      // std::cout << "------" << std::endl;
      // std::cout << "t_min: " << (st_idx_ptr->t_min & 0x00FFFFFFFFFFFFFF)
      //           << std::endl;

      // std::cout << "xid: " << (xid & 0x00FFFFFFFFFFFFFF) << std::endl;

      struct tpcc_stock *s_r =
          (struct tpcc_stock *)table_stock
              ->getVersions(st_idx_ptr->VID, st_idx_ptr->delta_id)
              ->get_readable_ver(xid);

      st_rec.s_quantity = s_r->s_quantity;
      st_rec.s_ytd = s_r->s_ytd;
      st_rec.s_order_cnt = s_r->s_order_cnt;
      st_rec.s_remote_cnt = s_r->s_remote_cnt;
    }

    // NOW UPDATE

    /*struct tpcc_stock *s_r = NULL;
    s_r = (struct tpcc_stock *)txn_op(ctask, hash_table, id, &op,
                                      ol_supply_w_id - 1);
    if (!s_r) {
      dprint("srv(%d): Aborting due to key %" PRId64 "\n", id, pkey);
      r = TXN_ABORT;
      goto final;
    }*/

    // update = s_ytd, s_order_cnt, s_remote_cnt, s_quantity

    short s_quantity = st_rec.s_quantity;
    st_rec.s_ytd += ol_quantity;
    st_rec.s_order_cnt++;
    if (remote) {
      st_rec.s_remote_cnt++;
    }

    short quantity;
    if (s_quantity > ol_quantity + 10)
      quantity = s_quantity - ol_quantity;
    else
      quantity = s_quantity - ol_quantity + 91;

    st_rec.s_quantity = quantity;

    table_stock->updateRecord(st_idx_ptr->VID, &st_rec, master_ver,
                              st_idx_ptr->last_master_ver, delta_ver,
                              st_idx_ptr->t_min, st_idx_ptr->t_max,
                              (xid >> 56) / NUM_CORE_PER_SOCKET, &st_col);

    st_idx_ptr->t_min = xid;
    st_idx_ptr->last_master_ver = master_ver;
    st_idx_ptr->delta_id = delta_ver;

    st_idx_ptr->latch.release();
  }

  /*
   * EXEC SQL UPDATE district SET d _next_o_id = :d _next_o_id + 1
   * WHERE d _id = :d_id AN D d _w _id = :w _id ;
   */
  std::vector<int> dist_col_scan = {8, 10};
  std::vector<int> dist_col_upd = {10};
  struct dist_read {
    float d_tax;
    uint64_t d_next_o_id;
  };
  struct dist_read dist_no_read;
  d_idx_ptr->latch.acquire();
  // std::cout << "D :" << dist_key << std::endl;
  if (txn::CC_MV2PL::is_readable(d_idx_ptr->t_min, d_idx_ptr->t_max, xid)) {
    table_district->getRecordByKey(d_idx_ptr->VID, d_idx_ptr->last_master_ver,
                                   &dist_col_scan, &dist_no_read);
  } else {
    // std::cout << "not readable 2" << std::endl;

    // std::cout << "t_min: " << d_idx_ptr->t_min << std::endl;
    // std::cout << "t_max: " << d_idx_ptr->t_max << std::endl;
    // std::cout << "xid: " << xid << std::endl;

    // std::cout << "------" << std::endl;
    // std::cout << "t_min: " << (d_idx_ptr->t_min & 0x00FFFFFFFFFFFFFF)
    //           << std::endl;

    // std::cout << "xid: " << (xid & 0x00FFFFFFFFFFFFFF) << std::endl;

    struct tpcc_district *d_r =
        (struct tpcc_district *)table_district
            ->getVersions(d_idx_ptr->VID, d_idx_ptr->delta_id)
            ->get_readable_ver(xid);

    dist_no_read.d_tax = d_r->d_tax;
    dist_no_read.d_next_o_id = d_r->d_next_o_id;
  }
  // NOW UPDATE

  uint64_t d_next_o_id_upd = dist_no_read.d_next_o_id + 1;
  table_district->updateRecord(
      d_idx_ptr->VID, &d_next_o_id_upd, master_ver, d_idx_ptr->last_master_ver,
      delta_ver, d_idx_ptr->t_min, d_idx_ptr->t_max,
      (xid >> 56) / NUM_CORE_PER_SOCKET, &dist_col_upd);

  d_idx_ptr->t_min = xid;
  d_idx_ptr->last_master_ver = master_ver;
  d_idx_ptr->delta_id = delta_ver;

  d_idx_ptr->latch.release();

  // TIME TO RELEASE LOCKS AS WE ARE NOT GONNA UPDATE ANYTHING
  txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired);

  //=============
  /*
   * EXEC SQL SELECT c_discount, c_last, c_credit, w_tax
   * INTO :c_discount, :c_last, :c_credit, :w_tax
   * FROM customer, warehouse
   * WHERE w_id = :w_id AND c_w_id = w_id AND c_d_id = :d_id AND c_id = :c_id;
   */

  // READ w_tax from warehouse_id where w_id

  // struct tpcc_warehouse *w_r =
  //   (struct tpcc_warehouse *)txn_op(ctask, hash_table, id, &op, w_id - 1);

  float w_tax = 0.0;
  std::vector<int> w_col_scan = {7};  // position in columns

  global_conf::IndexVal *w_idx_ptr =
      (global_conf::IndexVal *)table_warehouse->p_index->find(w_id);
  w_idx_ptr->latch.acquire();
  if (txn::CC_MV2PL::is_readable(w_idx_ptr->t_min, w_idx_ptr->t_max, xid)) {
    table_warehouse->getRecordByKey(w_idx_ptr->VID, w_idx_ptr->last_master_ver,
                                    &w_col_scan, &w_tax);
  } else {
    std::cout << "not readable 3" << std::endl;
    struct tpcc_warehouse *w_r =
        (struct tpcc_warehouse *)table_warehouse
            ->getVersions(w_idx_ptr->VID, w_idx_ptr->delta_id)
            ->get_readable_ver(xid);
    w_tax = w_r->w_tax;
  }
  w_idx_ptr->latch.release();

  //=============

  // READ c_discount, c_last, c_credit from customer where c_d_id, c_w_id, c_id,
  // TODO: copying the read values has issue, fix that

  // Here too, reading shit not required anywhere

  uint32_t cust_key = MAKE_CUST_KEY(w_id, d_id, c_id);
  std::vector<int> cust_col_scan = {5, 13, 15};

  struct cust_read {
    char c_last[LAST_NAME_LEN + 1];
    char c_credit[2];
    float c_discount;
  };
  struct cust_read cust_no_read;

  global_conf::IndexVal *c_idx_ptr =
      (global_conf::IndexVal *)table_customer->p_index->find(cust_key);
  assert(c_idx_ptr != NULL || c_idx_ptr != nullptr);

  c_idx_ptr->latch.acquire();
  if (txn::CC_MV2PL::is_readable(c_idx_ptr->t_min, c_idx_ptr->t_max, xid)) {
    table_customer->getRecordByKey(c_idx_ptr->VID, c_idx_ptr->last_master_ver,
                                   &cust_col_scan, &cust_no_read);

  } else {
    std::cout << "not readable 4" << std::endl;
    struct tpcc_customer *c_r =
        (struct tpcc_customer *)table_customer
            ->getVersions(c_idx_ptr->VID, c_idx_ptr->delta_id)
            ->get_readable_ver(xid);

    memcpy(cust_no_read.c_last, c_r->c_last, LAST_NAME_LEN + 1);
    memcpy(cust_no_read.c_credit, c_r->c_credit, 2);
    cust_no_read.c_discount = c_r->c_discount;
  }
  c_idx_ptr->latch.release();

  //=============

  /*
   * EXEC SQL INSERT IN TO ORDERS (o_id , o_d _id , o_w _id , o_c_id ,
   * o_entry_d , o_ol_cnt, o_all_local)
   * VALUES (:o_id , :d _id , :w _id , :c_id ,
   * :d atetime, :o_ol_cnt, :o_all_local);
   */
  uint64_t order_key = MAKE_CUST_KEY(w_id, d_id, dist_no_read.d_next_o_id);
  struct tpcc_order o_r;

  o_r.o_id = dist_no_read.d_next_o_id;
  o_r.o_c_id = c_id;
  o_r.o_d_id = d_id;
  o_r.o_w_id = w_id;
  o_r.o_entry_d = o_entry_d;
  o_r.o_ol_cnt = ol_cnt;

  // for now only local
  o_r.o_all_local = !remote;

  void *o_idx_ptr = table_order->insertRecord(&o_r, xid, master_ver);
  table_order->p_index->insert(order_key, o_idx_ptr);

  /*
   * EXEC SQL INSERT IN TO NEW_ORDER (no_o_id , no_d_id , no_w _id )
   * VALUES (:o_id , :d _id , :w _id );
   */

  struct tpcc_new_order no_r;

  no_r.no_o_id = dist_no_read.d_next_o_id;
  no_r.no_d_id = d_id;
  no_r.no_w_id = w_id;

  void *no_idx_ptr = table_new_order->insertRecord(&no_r, xid, master_ver);
  table_new_order->p_index->insert(order_key, no_idx_ptr);

  for (int ol_number = 0; ol_number < ol_cnt; ol_number++) {
    uint32_t ol_i_id = q->item[ol_number].ol_i_id;
    ushort ol_supply_w_id = q->item[ol_number].ol_supply_w_id;
    ushort ol_quantity = q->item[ol_number].ol_quantity;

    /*
     * EXEC SQL SELECT i_price, i_name , i_data
     * INTO :i_price, :i_name, :i_data
     * FROM item WHERE i_id = ol_i_id
     */

    float i_price;
    std::vector<int> i_col_scan = {3};

    global_conf::IndexVal *item_idx_ptr =
        (global_conf::IndexVal *)table_item->p_index->find(ol_i_id);
    item_idx_ptr->latch.acquire();
    if (txn::CC_MV2PL::is_readable(item_idx_ptr->t_min, item_idx_ptr->t_max,
                                   xid)) {
      table_item->getRecordByKey(item_idx_ptr->VID,
                                 item_idx_ptr->last_master_ver, &i_col_scan,
                                 &i_price);
    } else {
      std::cout << "not readable 5" << std::endl;
      struct tpcc_item *i_r =
          (struct tpcc_item *)table_item
              ->getVersions(item_idx_ptr->VID, item_idx_ptr->delta_id)
              ->get_readable_ver(xid);
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
    uint64_t ol_key =
        MAKE_OL_KEY(w_id, d_id, dist_no_read.d_next_o_id, ol_number);

    struct tpcc_order_line ol_ins;

    float ol_amount = ol_quantity * i_price * (1 + w_tax + dist_no_read.d_tax) *
                      (1 - cust_no_read.c_discount);

    ol_ins.ol_o_id = dist_no_read.d_next_o_id;
    ol_ins.ol_d_id = d_id;
    ol_ins.ol_w_id = w_id;
    ol_ins.ol_number = ol_number;
    ol_ins.ol_i_id = ol_i_id;
    ol_ins.ol_supply_w_id = ol_supply_w_id;
    // ol_ins.ol_delivery_d;  //
    ol_ins.ol_quantity = ol_quantity;
    ol_ins.ol_amount = ol_amount;
    // ol_ins.ol_dist_info[24];  //

    void *ol_idx_ptr = table_order_line->insertRecord(&ol_ins, xid, master_ver);
    table_order_line->p_index->insert(ol_key, ol_idx_ptr);

    /*MAKE_OP(op, OPTYPE_INSERT, (sizeof(struct tpcc_order_line)), pkey);
    struct tpcc_order_line *ol_r =
        (struct tpcc_order_line *)txn_op(ctask, hash_table, id, &op, w_id - 1);
    if (!ol_r) {
      printf("no_r w_id = %d d_id = %d o_id = %" PRId64 "\n", w_id, d_id, o_id);
    }
    assert(ol_r);
    dprint("srv(%d): inserted %" PRId64 "\n", id, pkey);*/
  }

  return true;
}
inline bool TPCC::exec_payment_txn(struct tpcc_query *stmts, uint64_t xid,
                                   ushort master_ver, ushort delta_ver) {
  return false;
}
inline bool TPCC::exec_orderstatus_txn(struct tpcc_query *stmts, uint64_t xid,
                                       ushort master_ver, ushort delta_ver) {
  return false;
}
inline bool TPCC::exec_delivery_txn(struct tpcc_query *stmts, uint64_t xid,
                                    ushort master_ver, ushort delta_ver) {
  return false;
}
inline bool TPCC::exec_stocklevel_txn(struct tpcc_query *stmts, uint64_t xid,
                                      ushort master_ver, ushort delta_ver) {
  return false;
}

inline void TPCC::tpcc_get_next_payment_query(int wid, void *arg) {
  // struct partition *p = &hash_table->partitions[s];
  struct tpcc_query *q = (struct tpcc_query *)arg;

  q->w_id = (wid) + 1;
  ;
  q->d_w_id = (wid) + 1;
  ;
  q->d_id = URand(&this->seed, 0, TPCC_NDIST_PER_WH - 1);
  q->h_amount = URand(&this->seed, 1, 5000);
  int x = URand(&this->seed, 1, 100);
  int y = URand(&this->seed, 1, 100);

  if (x <= 85 || g_dist_threshold == 100) {
    // home warehouse
    q->c_d_id = q->d_id;
    q->c_w_id = wid;
    ;
  } else {
    q->c_d_id = URand(&this->seed, 0, TPCC_NDIST_PER_WH);

    // remote warehouse if we have >1 wh
    if (num_warehouse > 1) {
      while ((q->c_w_id = URand(&this->seed, 0, num_warehouse - 1)) == wid)
        ;

    } else {
      q->c_w_id = wid;
      ;
    }
  }

  if (y <= 60) {
    // by last name
    q->by_last_name = TRUE;
    set_last_name(NURand(&this->seed, 255, 0, 999), q->c_last);
  } else {
    // by cust id
    q->by_last_name = FALSE;
    q->c_id = NURand(&this->seed, 1023, 0, TPCC_NCUST_PER_DIST - 1);
  }
}

void TPCC::tpcc_get_next_neworder_query(int wid, void *arg) {
  int ol_cnt, dup;
  struct tpcc_query *q = (struct tpcc_query *)arg;
  // std::cout << "WID IN GET: " << wid << std::endl;
  q->w_id = wid;

  // q->w_id = URand(&p->seed, 1, g_nservers);
  q->d_id = URand(&this->seed, 0, TPCC_NDIST_PER_WH - 1);
  // std::cout << "DID GET: " << q->d_id << std::endl;
  q->c_id = NURand(&this->seed, 1023, 0, TPCC_NCUST_PER_DIST - 1);
  q->rbk = URand(&this->seed, 1, 100);
  q->ol_cnt = URand(&this->seed, 5, TPCC_MAX_OL_PER_ORDER);
  q->o_entry_d = 2013;
  q->remote = 0;

  ol_cnt = q->ol_cnt;
  assert(ol_cnt <= TPCC_MAX_OL_PER_ORDER);
  for (int o = 0; o < ol_cnt; o++) {
    struct item *i = &q->item[o];

    do {
      i->ol_i_id = NURand(&this->seed, 8191, 0, TPCC_MAX_ITEMS - 1);

      // no duplicates
      dup = 0;
      for (int j = 0; j < o; j++)
        if (q->item[j].ol_i_id == i->ol_i_id) {
          dup = 1;
          break;
        }
    } while (dup);

    int x = URand(&this->seed, 0, 100);
    // if (g_dist_threshold == 100) x = 2;

    i->ol_supply_w_id = wid;

    // if (x > 1 || num_warehouse == 1) {
    //   // if (1) {
    //   i->ol_supply_w_id = wid;
    //   ;
    // } else {
    //   while ((i->ol_supply_w_id = URand(&this->seed, 0, num_warehouse - 1))
    //   ==
    //          q->w_id)
    //     ;

    //   q->remote = 1;
    // }
    assert(i->ol_supply_w_id < num_warehouse);

    i->ol_quantity = URand(&this->seed, 1, 10);
  }

  // print_tpcc_query(arg);
}

void TPCC::print_tpcc_query(void *arg) {
  struct tpcc_query *q = (struct tpcc_query *)arg;
  std::cout << "-------TPCC QUERY------" << std::endl;
  switch (q->query_type) {
    case NEW_ORDER:
      std::cout << "\tType: NEW_ORDER" << std::endl;
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

inline void TPCC::tpcc_get_next_orderstatus_query(int wid, void *arg) {
  struct tpcc_query *q = (struct tpcc_query *)arg;
  q->w_id = (wid) + 1;
  q->d_id = URand(&this->seed, 1, TPCC_NDIST_PER_WH);

  int y = URand(&this->seed, 1, 100);
  if (y <= 60) {
    // by last name
    q->by_last_name = TRUE;
    set_last_name(NURand(&this->seed, 255, 0, 999), q->c_last);
  } else {
    // by cust id
    q->by_last_name = FALSE;
    q->c_id = NURand(&this->seed, 1023, 1, TPCC_NCUST_PER_DIST);
  }
  q->c_w_id = (wid) + 1;
}

inline void TPCC::tpcc_get_next_delivery_query(int wid, void *arg) {
  struct tpcc_query *q = (struct tpcc_query *)arg;
  q->w_id = (wid) + 1;
  q->d_id = URand(&this->seed, 1, TPCC_NDIST_PER_WH);
  q->o_carrier_id = URand(&this->seed, 1, 10);
}

inline void TPCC::tpcc_get_next_stocklevel_query(int wid, void *arg) {
  struct tpcc_query *q = (struct tpcc_query *)arg;
  q->w_id = (wid) + 1;
  q->d_id = URand(&this->seed, 1, TPCC_NDIST_PER_WH);
  q->threshold = URand(&this->seed, 10, 20);
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
  // shuffle elements of the sequence array
  srand(time(NULL));
  for (int i = MIX_COUNT - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    TPCC_QUERY_TYPE temp = sequence[i];
    sequence[i] = sequence[j];
    sequence[j] = temp;
  }
}

TPCC::TPCC(std::string name, int num_warehouses, int g_dist_threshold, std::string csv_path,
       bool is_ch_benchmark)
    : Benchmark(name),
      num_warehouse(num_warehouses),
      g_dist_threshold(g_dist_threshold),
      csv_path(csv_path),
      is_ch_benchmark(is_ch_benchmark)  {
  this->schema = &storage::Schema::getInstance();
  this->seed = rand();

  uint64_t total_districts = TPCC_NDIST_PER_WH * this->num_warehouse;
  uint64_t max_customers = TPCC_NCUST_PER_DIST * total_districts;
  uint64_t max_orders = TPCC_MAX_ORDER_INITIAL_CAP;
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

  std::cout << "Total Memory Reserved for Tables: "
            << (double)this->schema->total_mem_reserved / (1024 * 1024 * 1024)
            << " GB" << std::endl;
  std::cout << "Total Memory Reserved for Deltas: "
            << (double)this->schema->total_delta_mem_reserved /
                   (1024 * 1024 * 1024)
            << " GB" << std::endl;
  cust_sec_index = new indexes::HashIndex<uint64_t, struct secondary_record>();
  cust_sec_index->reserve(max_customers);

  init_tpcc_seq_array();
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

/* A/C TPCC Specs*/
void TPCC::load_stock(int w_id) {
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
    void *hash_ptr = table_stock->insertRecord(stock_tmp, 0, 0);
    this->table_stock->p_index->insert(sid, hash_ptr);
  }

  delete stock_tmp;
}

/* A/C TPCC Specs*/
void TPCC::load_item() {
  // Primary Key: I_ID

  struct tpcc_item *item_temp = new struct tpcc_item;

  int orig[TPCC_MAX_ITEMS], pos;

  for (int i = 0; i < TPCC_MAX_ITEMS / 10; i++) orig[i] = 0;
  for (int i = 0; i < TPCC_MAX_ITEMS / 10; i++) {
    do {
      pos = URand(&this->seed, 0L, TPCC_MAX_ITEMS - 1);
    } while (orig[pos]);
    orig[pos] = 1;
  }

  for (uint32_t key = 0; key < TPCC_MAX_ITEMS; key++) {
    item_temp->i_id = key;
    item_temp->i_im_id = URand(&this->seed, 0L, TPCC_MAX_ITEMS - 1);

    make_alpha_string(&this->seed, 14, 24, item_temp->i_name);

    item_temp->i_price = ((float)URand(&this->seed, 100L, 10000L)) / 100.0;

    int data_len = make_alpha_string(&this->seed, 26, 50, item_temp->i_data);
    if (orig[key]) {
      int idx = URand(&this->seed, 0, data_len - 8);
      memcpy(&item_temp->i_data[idx], "original", 8);
    }

    // txn_id = 0, master_ver = 0
    void *hash_ptr = table_item->insertRecord(item_temp, 0, 0);
    this->table_item->p_index->insert(key, hash_ptr);
  }
  delete item_temp;
}

/* A/C TPCC Specs*/
void TPCC::load_district(int w_id) {
  // Primary Key: (D_W_ID, D_ID) D_W_ID
  // Foreign Key, references W_ID

  struct tpcc_district *r = new struct tpcc_district;

  for (int d = 0; d < TPCC_NDIST_PER_WH; d++) {
    ushort dkey = MAKE_DIST_KEY(w_id, d);
    r->d_id = d;
    r->d_w_id = w_id;

    make_alpha_string(&this->seed, 6, 10, r->d_name);
    make_alpha_string(&this->seed, 10, 20, r->d_street[0]);
    make_alpha_string(&this->seed, 10, 20, r->d_street[1]);
    make_alpha_string(&this->seed, 10, 20, r->d_city);
    make_alpha_string(&this->seed, 2, 2, r->d_state);
    make_alpha_string(&this->seed, 9, 9, r->d_zip);
    r->d_tax = (float)URand(&this->seed, 10L, 20L) / 100.0;
    r->d_ytd = 30000.0;
    r->d_next_o_id = 3000;

    void *hash_ptr = table_district->insertRecord(r, 0, 0);
    this->table_district->p_index->insert(dkey, hash_ptr);
  }

  delete r;
}

/* A/C TPCC Specs*/
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
  w_temp->w_tax = (float)URand(&this->seed, 10L, 20L) / 100.0;
  w_temp->w_ytd = 3000000.00;

  // txn_id = 0, master_ver = 0
  void *hash_ptr = table_warehouse->insertRecord(w_temp, 0, 0);
  this->table_warehouse->p_index->insert(w_id, hash_ptr);
  delete w_temp;
}

/* A/C TPCC Specs*/
void TPCC::load_history(int w_id) {
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

/* A/C TPCC Specs*/
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

  for (int d = 0; d < TPCC_NDIST_PER_WH; d++) {
    init_permutation(&this->seed, cperm);

    for (int o = 0; o < TPCC_NCUST_PER_DIST; o++) {
      uint32_t ckey = MAKE_ORDER_KEY(w_id, d, o);

      // e = hash_insert(p, key, sizeof(struct tpcc_order), NULL);
      // assert(e);

      // p->ninserts++;
      // e->ref_count++;

      // r = (struct tpcc_order *)e->value;

      int c_id = cperm[o];
      r->o_id = o;
      r->o_c_id = c_id;
      r->o_d_id = d;
      r->o_w_id = w_id;
      int o_entry = 2013;
      r->o_entry_d = 2013;
      if (o < 2100)
        r->o_carrier_id = URand(&this->seed, 1, 10);
      else
        r->o_carrier_id = 0;
      int o_ol_cnt = URand(&this->seed, 5, 15);
      r->o_ol_cnt = o_ol_cnt;
      r->o_all_local = 1;

      // insert order here
      void *hash_ptr = table_order->insertRecord(r, 0, 0);
      this->table_order->p_index->insert(ckey, hash_ptr);

      for (int ol = 0; ol < o_ol_cnt; ol++) {
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
        r_ol->ol_i_id = URand(&this->seed, 0, TPCC_MAX_ITEMS - 1);
        r_ol->ol_supply_w_id = w_id;

        if (o < 2100) {
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
      if (o >= 2100) {
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

/* A/C TPCC Specs*/
void TPCC::load_customer(int w_id) {
  // Primary Key: (C_W_ID, C_D_ID, C_ID)
  // (C_W_ID, C_D_ID) Foreign Key, references (D_W_ID, D_ID)

  // void *hash_ptr = table_customer->insertRecord(r, 0, 0);
  // this->table_customer->p_index->insert(key, hash_ptr);

  struct tpcc_customer *r = new tpcc_customer;
  for (int d = 0; d < TPCC_NDIST_PER_WH; d++) {
    for (int c = 0; c < TPCC_NCUST_PER_DIST; c++) {
      uint32_t ckey = MAKE_CUST_KEY(w_id, d, c);

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

      void *hash_ptr = table_customer->insertRecord(r, 0, 0);
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

void TPCC::load_data(int num_threads) {
  
  if (this->csv_path.size() > 1) {
      std::cout << "[TPCC] Load data from CSV: " << csv_path << std::endl;

      std::vector<std::thread> loaders;

      loaders.emplace_back([this]() { this->load_warehouse_csv(); });
      loaders.emplace_back([this]() { this->load_district_csv(); });
      loaders.emplace_back([this]() { this->load_stock_csv(); });
      loaders.emplace_back([this]() { this->load_customer_csv(); });
      loaders.emplace_back([this]() { this->load_history_csv(); });
      loaders.emplace_back([this]() { this->load_order_csv(); });
      loaders.emplace_back([this]() { this->load_neworder_csv(); });
      loaders.emplace_back([this]() { this->load_orderline_csv(); });
      loaders.emplace_back([this]() { this->load_supplier_csv(); });
      loaders.emplace_back([this]() { this->load_nation_csv(); });
      loaders.emplace_back([this]() { this->load_region_csv(); });
      loaders.emplace_back([this]() { this->load_item_csv(); });

      int i = 0;
      for (auto &th : loaders) {
        th.join();
      }

  } else{

    std::cout << "[TPCC] Load data" << std::endl;
    std::cout << "[TPCC] Loading Items: " << TPCC_MAX_ITEMS << std::endl;
    load_item();

    for (int w_id = 0; w_id < num_warehouse; w_id++) {
      std::cout << "[TPCC] Warehouse #" << w_id << " loading data..."
                << std::endl;

      load_warehouse(w_id);
      std::cout << "\t loading district..." << std::endl;
      load_district(w_id);
      std::cout << "\t loading stock..." << std::endl;
      load_stock(w_id);

      std::cout << "\t loading history..." << std::endl;
      load_history(w_id);
      std::cout << "\t loading customer..." << std::endl;
      load_customer(w_id);
      std::cout << "\t loading order..." << std::endl;
      load_order(w_id);
    }
  }
}


static std::string concat_path(const std::string &first, const std::string &second) {
  std::string ret = first;

  if (ret.back() != '/') {
    ret += "/";
  }
  ret += second;

  return ret;
}
// trim from start (in place)
static inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
  ltrim(s);
  rtrim(s);
}


// CSV Loaders

void TPCC::load_warehouse_csv(std::string filename, char delim) {
  /*
      path = ${csv_path}/WAREHOUSE.tbl
      csv schema:
      W_ID
      W_NAME
      W_STREET_1
      W_STREET_2
      W_CITY
      W_STATE
      W_ZIP
      W_TAX
      W_YTD
  */

  std::ifstream w_csv(concat_path(this->csv_path, filename).c_str());

  if (!w_csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;
  const uint num_fields = 9;

  uint field_cursor = 0;
  struct tpcc_warehouse w_temp;

  while (std::getline(w_csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;
    // std::cout << "-line-" << line << "-field#-" << field_cursor << std::endl;
    if (field_cursor == 1) {  // W_ID
      // std::cout << "--" << line << "--" << std::endl;
      w_temp.w_id = std::stoi(line, nullptr);

    } else if (field_cursor == 2) {  // W_NAME

      strncpy(w_temp.w_name, line.c_str(), line.length());
      w_temp.w_name[line.length()] = '\0';

    } else if (field_cursor == 3) {  // W_STREET_1
      strncpy(w_temp.w_street[0], line.c_str(), line.length());
      w_temp.w_street[0][line.length()] = '\0';

    } else if (field_cursor == 4) {  // W_STREET_2
      strncpy(w_temp.w_street[1], line.c_str(), line.length());
      w_temp.w_street[1][line.length()] = '\0';

    } else if (field_cursor == 5) {  // W_CITY
      strncpy(w_temp.w_city, line.c_str(), line.length());
      w_temp.w_city[line.length()] = '\0';

    } else if (field_cursor == 6) {  // W_STATE (fixed-length)
      strncpy(w_temp.w_state, line.c_str(), line.length());

    } else if (field_cursor == 7) {  // W_ZIP (fixed-length)
      strncpy(w_temp.w_zip, line.c_str(), line.length());

    } else if (field_cursor == 8) {  // W_TAX

      w_temp.w_tax = std::stof(line, nullptr);

    } else if (field_cursor == 9) {  // W_YTD

      w_temp.w_ytd = std::stof(line, nullptr);

      //(field_cusor == num_fields)
      // insert record
      void *hash_ptr = table_warehouse->insertRecord(&w_temp, 0, 0);
      this->table_warehouse->p_index->insert(w_temp.w_id, hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }

    //   struct tpcc_warehouse {
    //   short w_id;
    //   char w_name[11];
    //   char w_street[2][21];
    //   char w_city[21];
    //   char w_state[2];
    //   char w_zip[9];
    //   float w_tax;
    //   float w_ytd;
    // };
  }
  std::cout << "\t loaded warehouse: " << n_records << std::endl;

  w_csv.close();
}

void TPCC::load_stock_csv(std::string filename, char delim) {
  /*
      csv schema:
        S_I_ID
        S_W_ID
        S_QUANTITY
        S_DIST_01.  4
        S_DIST_02
        S_DIST_03
        S_DIST_04
        S_DIST_05
        S_DIST_06
        S_DIST_07
        S_DIST_08
        S_DIST_09
        S_DIST_10.  13
        S_YTD
        S_ORDER_CNT
        S_REMOTE_CNT
        S_DATA
        S_SU_SUPPKEY - no TPC-C/CH-benCHmark spec
  */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_stock temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {  // S_I_ID
      temp.s_i_id = std::stoi(line, nullptr);

    } else if (field_cursor == 2) {  // S_W_ID

      temp.s_w_id = std::stoi(line, nullptr);

    } else if (field_cursor == 3) {  // S_QUANTITY

      temp.s_quantity = (short)std::stoi(line, nullptr);

    } else if (field_cursor >= 4 &&
               field_cursor <= 13) {  // S_DIST_01 - S_DIST_10

      strncpy(temp.s_dist[field_cursor - 4], line.c_str(),
              line.length());  // fixed text, size 23

    } else if (field_cursor == 14) {  // S_YTD

      temp.s_ytd = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 15) {  // S_ORDER_CNT

      temp.s_order_cnt = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 16) {  // S_REMOTE_CNT

      temp.s_remote_cnt = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 17) {  // S_DATA

      strncpy(temp.s_data, line.c_str(), line.length());
      temp.s_data[line.length()] = '\0';

    } else if (field_cursor == 18) {
      // S_SU_SUPPKEY - no TPC-C/CH-benCHmark spec
      temp.s_su_suppkey = std::stoi(line, nullptr);
    }

    if ((!is_ch_benchmark && field_cursor == 17) ||
        (is_ch_benchmark && field_cursor == 18)) {
      // insert record
      void *hash_ptr = table_stock->insertRecord(&temp, 0, 0);
      this->table_stock->p_index->insert(
          MAKE_STOCK_KEY(temp.s_w_id, temp.s_i_id), hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded stock: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_item_csv(std::string filename, char delim) {
  /*
        csv schema:
          uint32_t i_id;
          uint32_t i_im_id;
          char i_name[25];
          float i_price;
          char i_data[51];
    */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_item temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {  // I_ID
      temp.i_id = std::stoi(line, nullptr);

    } else if (field_cursor == 2) {  // I_IM_ID

      temp.i_im_id = std::stoi(line, nullptr);

    } else if (field_cursor == 3) {  // I_NAME

      strncpy(temp.i_name, line.c_str(), line.length());

    } else if (field_cursor == 4) {  // I_PRICE

      temp.i_price = std::stof(line, nullptr);

    } else if (field_cursor == 5) {  // I_DATA

      strncpy(temp.i_data, line.c_str(), line.length());
      temp.i_data[line.length()] = '\0';

      // insert record
      void *hash_ptr = table_item->insertRecord(&temp, 0, 0);
      this->table_item->p_index->insert(temp.i_id, hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded items: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_district_csv(std::string filename, char delim) {
  /*
        csv schema:
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
    */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_district temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {  // D_ID
      temp.d_id = std::stoi(line, nullptr);

    } else if (field_cursor == 2) {
      temp.d_w_id = std::stoi(line, nullptr);

    } else if (field_cursor == 3) {
      strncpy(temp.d_name, line.c_str(), line.length());
      temp.d_name[line.length()] = '\0';

    } else if (field_cursor == 4) {
      strncpy(temp.d_street[0], line.c_str(), line.length());
      temp.d_street[0][line.length()] = '\0';

    } else if (field_cursor == 5) {
      strncpy(temp.d_street[1], line.c_str(), line.length());
      temp.d_street[1][line.length()] = '\0';

    } else if (field_cursor == 6) {
      strncpy(temp.d_city, line.c_str(), line.length());
      temp.d_city[line.length()] = '\0';

    } else if (field_cursor == 7) {
      strncpy(temp.d_state, line.c_str(), line.length());

    } else if (field_cursor == 8) {
      strncpy(temp.d_zip, line.c_str(), line.length());

    } else if (field_cursor == 9) {
      temp.d_tax = std::stof(line, nullptr);

    } else if (field_cursor == 10) {
      temp.d_ytd = std::stof(line, nullptr);

    } else if (field_cursor == 11) {
      temp.d_next_o_id = std::stoi(line, nullptr);

      // insert record
      void *hash_ptr = table_district->insertRecord(&temp, 0, 0);
      bool done = this->table_district->p_index->insert(
          MAKE_DIST_KEY(temp.d_w_id, temp.d_id), hash_ptr);

      assert(done);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded districts: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_nation_csv(std::string filename, char delim) {
  /*
        csv schema:
          ushort n_nationkey;
          char n_name[16];  // var
          ushort n_regionkey;
          char n_comment[115];  // var
    */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct ch_nation temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {
      temp.n_nationkey = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 2) {
      strncpy(temp.n_name, line.c_str(), line.length());
      temp.n_name[line.length()] = '\0';

    } else if (field_cursor == 3) {
      temp.n_regionkey = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 4) {
      strncpy(temp.n_comment, line.c_str(), line.length());
      temp.n_comment[line.length()] = '\0';

      // insert record
      void *hash_ptr = table_nation->insertRecord(&temp, 0, 0);
      this->table_nation->p_index->insert(temp.n_nationkey, hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded nations: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_region_csv(std::string filename, char delim) {
  /*
        csv schema:
         ushort r_regionkey;
    char r_name[12];      // var
    char r_comment[115];  // var
    */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct ch_region temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {
      temp.r_regionkey = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 2) {
      strncpy(temp.r_name, line.c_str(), line.length());
      temp.r_name[line.length()] = '\0';

    } else if (field_cursor == 3) {
      strncpy(temp.r_comment, line.c_str(), line.length());
      temp.r_comment[line.length()] = '\0';

      // insert record
      void *hash_ptr = table_region->insertRecord(&temp, 0, 0);
      this->table_region->p_index->insert(temp.r_regionkey, hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded regions: " << n_records << std::endl;

  csv.close();
}

void TPCC::load_supplier_csv(std::string filename, char delim) {
  /*
     csv schema:
       uint32_t suppkey;
       char s_name[18];     // fix
       char s_address[41];  // var
       ushort s_nationkey;
       char s_phone[15];  // fix
       float s_acctbal;
       char s_comment[101];  // var
 */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct ch_supplier temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {
      temp.suppkey = std::stoi(line, nullptr);

    } else if (field_cursor == 2) {
      strncpy(temp.s_name, line.c_str(), line.length());

    } else if (field_cursor == 3) {
      strncpy(temp.s_address, line.c_str(), line.length());
      temp.s_address[line.length()] = '\0';

    } else if (field_cursor == 4) {
      temp.s_nationkey = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 5) {
      strncpy(temp.s_phone, line.c_str(), line.length());

    } else if (field_cursor == 6) {
      temp.s_acctbal = std::stof(line, nullptr);

    } else if (field_cursor == 7) {
      strncpy(temp.s_comment, line.c_str(), line.length());
      temp.s_comment[line.length()] = '\0';

      // insert record
      void *hash_ptr = table_supplier->insertRecord(&temp, 0, 0);
      this->table_supplier->p_index->insert(temp.suppkey, hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded supplier: " << n_records << std::endl;

  csv.close();
}

void TPCC::load_neworder_csv(std::string filename, char delim) {
  /*
    uint64_t no_o_id;
    ushort no_d_id;
    ushort no_w_id;
    uint64_t order_key = MAKE_CUST_KEY(w_id, d_id, dist_no_read.d_next_o_id);
  */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_new_order temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {
      temp.no_o_id = std::stoull(line, nullptr);

    } else if (field_cursor == 2) {
      temp.no_d_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 3) {
      temp.no_w_id = (ushort)std::stoi(line, nullptr);

      // insert record
      void *hash_ptr = table_new_order->insertRecord(&temp, 0, 0);
      this->table_new_order->p_index->insert(
          MAKE_CUST_KEY(temp.no_w_id, temp.no_d_id, temp.no_o_id), hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded new orders: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_history_csv(std::string filename, char delim) {
  /*
   uint32_t h_c_id;
    ushort h_c_d_id;
    ushort h_c_w_id;
    ushort h_d_id;
    ushort h_w_id;
    uint32_t h_date;
    float h_amount;
    char h_data[25];
  */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_history temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {
      temp.h_c_id = std::stoi(line, nullptr);

    } else if (field_cursor == 2) {
      temp.h_c_d_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 3) {
      temp.h_c_w_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 4) {
      temp.h_d_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 5) {
      temp.h_w_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 6) {
      temp.h_date = std::stoi(line, nullptr);

    } else if (field_cursor == 7) {
      temp.h_amount = std::stof(line, nullptr);

    } else if (field_cursor == 8) {
      strncpy(temp.h_data, line.c_str(), line.length());
      temp.h_data[line.length()] = '\0';

      // insert record
      void *hash_ptr = table_history->insertRecord(&temp, 0, 0);
      this->table_history->p_index->insert(
          MAKE_CUST_KEY(temp.h_w_id, temp.h_d_id, temp.h_c_id), hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded history: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_order_csv(std::string filename, char delim) {
  /*
   uint64_t o_id;
    ushort o_d_id;
    ushort o_w_id;
    uint32_t o_c_id;
    uint32_t o_entry_d;
    short o_carrier_id;
    ushort o_ol_cnt;
    ushort o_all_local;
  */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_order temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;

    if (field_cursor == 1) {
      temp.o_id = std::stoull(line, nullptr);

    } else if (field_cursor == 2) {
      temp.o_d_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 3) {
      temp.o_w_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 4) {
      temp.o_c_id = std::stoi(line, nullptr);

    } else if (field_cursor == 5) {
      temp.o_entry_d = std::stoi(line, nullptr);

    } else if (field_cursor == 6) {
      temp.o_carrier_id = (short)std::stoi(line, nullptr);

    } else if (field_cursor == 7) {
      temp.o_ol_cnt = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 8) {
      temp.o_all_local = (ushort)std::stoi(line, nullptr);

      // insert record
      void *hash_ptr = table_order->insertRecord(&temp, 0, 0);
      this->table_order->p_index->insert(
          MAKE_ORDER_KEY(temp.o_w_id, temp.o_d_id, temp.o_id), hash_ptr);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded orders: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_orderline_csv(std::string filename, char delim) {
  /*
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
  */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_order_line temp;

  while (std::getline(csv, line, delim)) {
    try {
      // trim(line);
      // if (line.length() == 0) continue;
      if (line == "\n" || line == "\r\n") continue;

      field_cursor++;
      // std::cout << "line--" << line << "--cursor--" << field_cursor << "-"
      //         << n_records << std::endl;

      if (field_cursor == 1) {
        temp.ol_o_id = std::stoull(line, nullptr);

      } else if (field_cursor == 2) {
        temp.ol_d_id = (ushort)std::stoi(line, nullptr);

      } else if (field_cursor == 3) {
        temp.ol_w_id = (ushort)std::stoi(line, nullptr);

      } else if (field_cursor == 4) {
        temp.ol_number = (ushort)std::stoi(line, nullptr);

      } else if (field_cursor == 5) {
        temp.ol_i_id = (ushort)std::stoi(line, nullptr);

      } else if (field_cursor == 6) {
        temp.ol_supply_w_id = (ushort)std::stoi(line, nullptr);

      } else if (field_cursor == 7) {
        if (line.length() != 0) {
          temp.ol_delivery_d = std::stoi(line, nullptr);
        } else {
          temp.ol_delivery_d = 0;
        }

      } else if (field_cursor == 8) {
        temp.ol_quantity = (short)std::stoi(line, nullptr);

      } else if (field_cursor == 9) {
        temp.ol_amount = std::stof(line, nullptr);

      } else if (field_cursor == 10) {
        strncpy(temp.ol_dist_info, line.c_str(), line.length());

        // insert record
        void *hash_ptr = table_order_line->insertRecord(&temp, 0, 0);
        this->table_order_line->p_index->insert(
            MAKE_OL_KEY(temp.ol_w_id, temp.ol_d_id, temp.ol_o_id,
                        temp.ol_number),
            hash_ptr);

        // reset cursor
        field_cursor = 0;
        n_records++;
      }
    /* */ } catch (...) {
      /* */
      std::cout << "Expception in line--" << line << "--field-cursor-"
                << field_cursor << "--nrec--" << n_records << std::endl;
      exit(1);
    }
  }
  std::cout << "\t loaded orderline: " << n_records << std::endl;

  csv.close();
}
void TPCC::load_customer_csv(std::string filename, char delim) {
  /*
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

    secondary index too
    */

  std::ifstream csv(concat_path(this->csv_path, filename).c_str());

  if (!csv.is_open()) {
    assert(false);
  }

  std::string line;
  uint64_t n_records = 0;

  uint field_cursor = 0;
  struct tpcc_customer temp;

  while (std::getline(csv, line, delim)) {
    trim(line);
    if (line.length() == 0) continue;

    field_cursor++;
    // std::cout << "LINE:-" << line << "--" << field_cursor << std::endl;

    if (field_cursor == 1) {
      temp.c_id = std::stoi(line, nullptr);

    } else if (field_cursor == 2) {
      temp.c_d_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 3) {
      temp.c_w_id = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 4) {
      strncpy(temp.c_first, line.c_str(), line.length());
      temp.c_first[line.length()] = '\0';

    } else if (field_cursor == 5) {
      strncpy(temp.c_middle, line.c_str(), line.length());

    } else if (field_cursor == 6) {
      strncpy(temp.c_last, line.c_str(), line.length());
      temp.c_last[line.length()] = '\0';

    } else if (field_cursor == 7) {
      strncpy(temp.c_street[0], line.c_str(), line.length());
      temp.c_street[0][line.length()] = '\0';

    } else if (field_cursor == 8) {
      strncpy(temp.c_street[1], line.c_str(), line.length());
      temp.c_street[1][line.length()] = '\0';

    } else if (field_cursor == 9) {
      strncpy(temp.c_city, line.c_str(), line.length());
      temp.c_city[line.length()] = '\0';

    } else if (field_cursor == 10) {
      strncpy(temp.c_state, line.c_str(), line.length());

    } else if (field_cursor == 11) {
      strncpy(temp.c_zip, line.c_str(), line.length());

    } else if (field_cursor == 12) {
      strncpy(temp.c_phone, line.c_str(), line.length());

    } else if (field_cursor == 13) {
      temp.c_since = 123456789;  //(uint32_t)std::stoull(line, nullptr);

    } else if (field_cursor == 14) {
      strncpy(temp.c_credit, line.c_str(), line.length());

    } else if (field_cursor == 15) {
      temp.c_credit_lim = std::stof(line, nullptr);

    } else if (field_cursor == 16) {
      temp.c_discount = std::stof(line, nullptr);

    } else if (field_cursor == 17) {
      temp.c_balance = std::stof(line, nullptr);

    } else if (field_cursor == 18) {
      temp.c_ytd_payment = std::stof(line, nullptr);

    } else if (field_cursor == 19) {
      temp.c_payment_cnt = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 20) {
      temp.c_delivery_cnt = (ushort)std::stoi(line, nullptr);

    } else if (field_cursor == 21) {
      strncpy(temp.c_data, line.c_str(), line.length());
      temp.c_data[line.length()] = '\0';

    } else if (field_cursor == 22) {
      temp.c_n_nationkey = (ushort)std::stoi(line, nullptr);

      // insert record
      void *hash_ptr = table_customer->insertRecord(&temp, 0, 0);
      this->table_customer->p_index->insert(
          MAKE_CUST_KEY(temp.c_w_id, temp.c_d_id, temp.c_id), hash_ptr);

      load_customer_secondary_index(temp);

      // reset cursor
      field_cursor = 0;
      n_records++;
    }
  }
  std::cout << "\t loaded customers: " << n_records << std::endl;

  csv.close();
}

void TPCC::load_customer_secondary_index(struct tpcc_customer &r) {
  uint64_t sr_dkey = cust_derive_key(r.c_last, r.c_d_id, r.c_w_id);

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
    sr.sr_rids = (uint32_t *)realloc(sr.sr_rids, sizeof(uint64_t) * sr_nids);
    assert(sr.sr_rids);
  }

  sr.sr_idx = sr_idx;
  sr.sr_nids = sr_nids;

  cust_sec_index->update(sr_dkey, sr);
}

}  // namespace bench
