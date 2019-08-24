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

#include "micro_ssb.hpp"

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

/*

  Note: There are aborts when reading from CSV, there is something wrong
  there for sure. the issue intially was that every ID was starting
  from 1 for the chBenchmark generator.


  Optimize data types, sometimes it is uint and sometimes it is short.
*/

namespace bench {

static inline uint32_t get_timestamp() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

static std::string concat_path(const std::string &first,
                               const std::string &second) {
  std::string ret = first;

  if (ret.back() != '/') {
    ret += "/";
  }
  ret += second;

  return ret;
}

bool MicroSSB::exec_txn(void *stmts, uint64_t xid, ushort master_ver,
                        ushort delta_ver) {
  struct ssb_query *q = (struct ssb_query *)stmts;
  int ol_cnt = q->ol_cnt;
  uint32_t custkey = q->custkey;
  uint32_t suppkey = q->suppkey;

  std::vector<global_conf::IndexVal *> hash_ptrs_lock_acquired;

  // acquire lock on all part keys
  for (int ol_number = 0; ol_number < ol_cnt; ol_number++) {
    uint32_t partkey = q->parts[ol_number].partkey;

    // ACQUIRE WRITE_LOCK FOR DISTRICT
    global_conf::IndexVal *part_idx_ptr =
        (global_conf::IndexVal *)table_part->p_index->find(partkey);

    assert(part_idx_ptr != NULL || part_idx_ptr != nullptr);
    bool e_false_s = false;
    if (part_idx_ptr->write_lck.compare_exchange_strong(e_false_s, true)) {
      hash_ptrs_lock_acquired.emplace_back(part_idx_ptr);
      q->parts[ol_number].idx_ptr = (void *)part_idx_ptr;
    } else {
      // ABORT
      txn::CC_MV2PL::release_locks(hash_ptrs_lock_acquired);
      return false;
    }
  }

  // update part key
  // insert into lineorder (insert together as ssb is denormalized)

  std::vector<ushort> pt_col = {1};

  for (int ol_number = 0; ol_number < ol_cnt; ol_number++) {
    uint32_t partkey = q->parts[ol_number].partkey;
    uint ol_quantity = q->parts[ol_number].quantity;

    uint32_t p_stocklevel;

    // update part table

    global_conf::IndexVal *part_idx_ptr =
        (global_conf::IndexVal *)q->parts[ol_number].idx_ptr;

    assert(part_idx_ptr != NULL || part_idx_ptr != nullptr);
    part_idx_ptr->latch.acquire();

    if (txn::CC_MV2PL::is_readable(part_idx_ptr->t_min, part_idx_ptr->t_max,
                                   xid)) {
      table_part->getRecordByKey(part_idx_ptr->VID,
                                 part_idx_ptr->last_master_ver, &pt_col,
                                 &p_stocklevel);
    } else {
      // std::cout << "not readable 1" << std::endl;
      // std::cout << "t_min: " << st_idx_ptr->t_min << std::endl;
      // std::cout << "t_max: " << st_idx_ptr->t_max << std::endl;
      // std::cout << "xid: " << xid << std::endl;

      // std::cout << "------" << std::endl;
      // std::cout << "t_min: " << (st_idx_ptr->t_min & 0x00FFFFFFFFFFFFFF)
      //           << std::endl;

      // std::cout << "xid: " << (xid & 0x00FFFFFFFFFFFFFF) << std::endl;

      p_stocklevel =
          ((struct part *)table_part
               ->getVersions(part_idx_ptr->VID, part_idx_ptr->delta_id)
               ->get_readable_ver(xid))
              ->p_stocklevel;
    }

    // NOW UPDATE

    uint32_t quantity;
    if (p_stocklevel > ol_quantity + 10)
      quantity = p_stocklevel - ol_quantity;
    else
      quantity = p_stocklevel - ol_quantity + 91;

    p_stocklevel = quantity;

    table_part->updateRecord(part_idx_ptr->VID, &p_stocklevel, master_ver,
                             part_idx_ptr->last_master_ver, delta_ver,
                             part_idx_ptr->t_min, part_idx_ptr->t_max,
                             (xid >> 56) / NUM_CORE_PER_SOCKET, &pt_col);

    part_idx_ptr->t_min = xid;
    part_idx_ptr->last_master_ver = master_ver;
    part_idx_ptr->delta_id = delta_ver;
    part_idx_ptr->write_lck.store(false);
    part_idx_ptr->latch.release();

    // insert lineorder
  }

  return true;
}
void MicroSSB::gen_txn(int wid, void *txn_ptr) {
  struct ssb_query *q = (struct ssb_query *)txn_ptr;
  int dup;
  /*
    struct ssb_query_part {
      uint32_t partkey;
      uint32_t quantity;
    };
    struct ssb_query {
      uint32_t custkey;
      uint32_t suppkey;
      uint32_t ol_cnt;
      struct part parts[MICRO_SBB_MAX_PART_PER_ORDER];

      // uint32_t orderpriority;
      // uint32_t shippriority;
    };
  */

  q->custkey = NURand(&this->seed, 1023, 0, NUM_CUSTOMERS - 1);
  q->suppkey = URand(&this->seed, 0, NUM_SUPPLIERS - 1);

  // Parts
  q->ol_cnt = URand(&this->seed, 5, MICRO_SSB_MAX_PART_PER_ORDER);
  for (int o = 0; o < q->ol_cnt; o++) {
    struct ssb_query_part *i = &q->parts[o];

    do {
      i->partkey = NURand(&this->seed, 8191, 0, NUM_PARTS - 1);

      // no duplicates
      dup = 0;
      for (int j = 0; j < o; j++)
        if (q->parts[j].partkey == i->partkey) {
          dup = 1;
          break;
        }
    } while (dup);

    i->quantity = URand(&this->seed, 1, 10);
    assert(i->partkey < NUM_PARTS);
  }
}

void MicroSSB::load_lineorder(uint64_t num_lineorder) {
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_orderkey",
          concat_path(this->data_path, "lineorder.csv.lo_orderkey").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_linenumber",
          concat_path(this->data_path, "lineorder.csv.lo_linenumber").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_custkey",
          concat_path(this->data_path, "lineorder.csv.lo_custkey").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_partkey",
          concat_path(this->data_path, "lineorder.csv.lo_partkey").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_suppkey",
          concat_path(this->data_path, "lineorder.csv.lo_suppkey").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_orderdate",
          concat_path(this->data_path, "lineorder.csv.lo_orderdate").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_quantity",
          concat_path(this->data_path, "lineorder.csv.lo_quantity").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_extendedprice",
          concat_path(this->data_path, "lineorder.csv.lo_extendedprice")
              .c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_ordtotalprice",
          concat_path(this->data_path, "lineorder.csv.lo_ordtotalprice")
              .c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_discount",
          concat_path(this->data_path, "lineorder.csv.lo_discount").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_revenue",
          concat_path(this->data_path, "lineorder.csv.lo_revenue").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_supplycost",
          concat_path(this->data_path, "lineorder.csv.lo_supplycost").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_tax",
          concat_path(this->data_path, "lineorder.csv.lo_tax").c_str());
  ((storage::ColumnStore *)this->table_lineorder)
      ->load_data_from_binary(
          "lo_commitdate",
          concat_path(this->data_path, "lineorder.csv.lo_commitdate").c_str());

  for (uint64_t i = 0; i < num_lineorder; i++) {
    this->table_part->insertIndexRecord(0, 0);
  }
}
void MicroSSB::load_part(uint64_t num_parts) {
  // insert records

  // p_partkey
  std::cout << "loading p_partkey from "
            << concat_path(this->data_path, "part.csv.p_partkey") << std::endl;
  ((storage::ColumnStore *)this->table_part)
      ->load_data_from_binary(
          "p_partkey",
          concat_path(this->data_path, "part.csv.p_partkey").c_str());
  // p_size
  std::cout << "loading p_size from "
            << concat_path(this->data_path, "part.csv.p_size") << std::endl;
  ((storage::ColumnStore *)this->table_part)
      ->load_data_from_binary(
          "p_size", concat_path(this->data_path, "part.csv.p_size").c_str());
  // // p_stocklevel
  ((storage::ColumnStore *)this->table_part)
      ->load_data_from_binary(
          "p_stocklevel",
          concat_path(this->data_path, "part.csv.p_stocklevel").c_str());

  std::cout << "Inserting index: " << num_parts << std::endl;
  for (uint64_t i = 0; i < num_parts; i++) {
    this->table_part->insertIndexRecord(0, 0);
  }
  // void *hash_ptr = table_item->insertRecord(&temp, 0, 0);
  // assert(this->table_item->p_index->insert(temp.i_id, hash_ptr));
}

void MicroSSB::create_tbl_lineorder(uint64_t num_lineorder) {
  /*
  struct lineorder {
      uint32_t lo_orderkey;
      uint32_t lo_linenumber;
      uint32_t lo_custkey;
      uint32_t lo_partkey;
      uint32_t lo_suppkey;
      uint32_t lo_orderdate;
      char lo_orderpriority[12];
      char lo_shippriority[12];
      uint32_t lo_quantity;
      uint32_t lo_extendedprice;
      uint32_t lo_ordtotalprice;
      uint32_t lo_discount;
      uint32_t lo_revenue;
      uint32_t lo_supplycost;
      uint32_t lo_tax;
      uint32_t lo_commitdate;
      char lo_shipmode[12];
    };
  */

  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  struct lineorder tmp;

  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_orderkey", storage::INTEGER, sizeof(tmp.lo_orderkey)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_linenumber", storage::INTEGER, sizeof(tmp.lo_linenumber)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_custkey", storage::INTEGER, sizeof(tmp.lo_custkey)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_partkey", storage::INTEGER, sizeof(tmp.lo_partkey)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_suppkey", storage::INTEGER, sizeof(tmp.lo_suppkey)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_orderdate", storage::INTEGER, sizeof(tmp.lo_orderdate)));
  // columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
  //     "lo_orderpriority", storage::STRING, sizeof(tmp.lo_orderpriority)));
  // columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
  //     "lo_shippriority", storage::STRING, sizeof(tmp.lo_shippriority)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_quantity", storage::INTEGER, sizeof(tmp.lo_quantity)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_extendedprice", storage::INTEGER, sizeof(tmp.lo_extendedprice)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_ordtotalprice", storage::INTEGER, sizeof(tmp.lo_ordtotalprice)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_discount", storage::INTEGER, sizeof(tmp.lo_discount)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_revenue", storage::INTEGER, sizeof(tmp.lo_revenue)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_supplycost", storage::INTEGER, sizeof(tmp.lo_supplycost)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_tax", storage::INTEGER, sizeof(tmp.lo_tax)));

  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_commitdate", storage::INTEGER, sizeof(tmp.lo_commitdate)));
  // columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
  //     "lo_shipmode", storage::STRING, sizeof(tmp.lo_shipmode)));

  table_lineorder =
      schema->create_table("ssbm_lineorder", storage::COLUMN_STORE, columns,
                           num_lineorder + LINEORDER_EXTRA_RESERVE);
}

void MicroSSB::create_tbl_part(uint64_t num_parts) {
  /*
  struct part {
    uint32_t p_partkey;
    uint32_t p_stocklevel;  // added for upds.
    char p_name[12];
    char p_mfgr[12];
    char p_category[12];
    char p_brand1[12];
    char p_color[12];
    char p_type[12];
    uint32_t p_size[12];
    char p_container[12];
  };
  */
  std::vector<std::tuple<std::string, storage::data_type, size_t>> columns;

  struct part tmp;

  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_partkey", storage::INTEGER, sizeof(tmp.p_partkey)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_stocklevel", storage::INTEGER, sizeof(tmp.p_stocklevel)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_size", storage::INTEGER, sizeof(tmp.p_size)));

  /*
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_name", storage::STRING, 25));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_mfgr", storage::STRING, 25));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_category", storage::STRING, 25));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_brand1", storage::STRING, 25));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_color", storage::STRING, 25));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_type", storage::STRING, 25));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "p_container", storage::STRING, 25));*/

  table_part = schema->create_table("ssbm_part", storage::COLUMN_STORE, columns,
                                    num_parts + 10);
}

MicroSSB::~MicroSSB() {}
MicroSSB::MicroSSB(std::string name, std::string binary_path_root)
    : Benchmark(name), data_path(binary_path_root) {
  std::cout << "Benchmark Init: " << name << std::endl;
  this->schema = &storage::Schema::getInstance();
  this->seed = rand();
  create_tbl_part();
  create_tbl_lineorder();

  std::cout << "Total Memory Reserved for Tables: "
            << (double)this->schema->total_mem_reserved / (1024 * 1024 * 1024)
            << " GB" << std::endl;
  std::cout << "Total Memory Reserved for Deltas: "
            << (double)this->schema->total_delta_mem_reserved /
                   (1024 * 1024 * 1024)
            << " GB" << std::endl;
}
void MicroSSB::load_data(int num_threads) {
  std::cout << "[TPCC] Load data from : " << data_path << std::endl;
  std::vector<std::thread> loaders;

  loaders.emplace_back([this]() { this->load_lineorder(); });
  loaders.emplace_back([this]() { this->load_part(); });

  int i = 0;
  for (auto &th : loaders) {
    th.join();
  }
}

};  // namespace bench
