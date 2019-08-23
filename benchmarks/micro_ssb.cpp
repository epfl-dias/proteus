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

void MicroSSB::get_next_neworder_query(void *arg) {}
bool MicroSSB::exec_txn(void *stmts, uint64_t xid, ushort master_ver,
                        ushort delta_ver) {
  return true;
}
void MicroSSB::gen_txn(int wid, void *txn_ptr) {}

void MicroSSB::load_data(int num_threads) {
  std::cout << "[TPCC] Load data from : " << data_path << std::endl;
  std::vector<std::thread> loaders;

  // loaders.emplace_back([this]() { this->load_lineorder(); });
  loaders.emplace_back([this]() { this->load_part(); });

  int i = 0;
  for (auto &th : loaders) {
    th.join();
  }
}
void MicroSSB::load_lineorder(uint64_t num_lineorder) {
  // std::ifstream binFile(concat_path(this->csv_path, filename).c_str(),
  // std::ifstream::binary); if(binFile) {
  //   // get length of file
  //   binFile.seekg(0, binFile.end);
  //   size_t length = static_cast<size_t>(binFile.tellg());
  //   binFile.seekg(0, binFile.beg);

  //   // read whole contents of the file to a buffer at once
  //   char *buffer = new char[length];
  //   binFile.read(buffer, length);
  //   binFile.close();

  for (uint64_t i = 0; i < num_lineorder; i++) {
    this->table_part->insertIndexRecord(0, 0);
  }
}
void MicroSSB::load_part(uint64_t num_parts) {
  // map the column..
  //

  // p_stocklevel .. URand(&this->seed, 1000, 10000);

  // std::ifstream binFile(concat_path(this->csv_path, filename).c_str(),
  // std::ifstream::binary); if(binFile) {
  //   // get length of file
  //   binFile.seekg(0, binFile.end);
  //   size_t length = static_cast<size_t>(binFile.tellg());
  //   binFile.seekg(0, binFile.beg);

  //   // read whole contents of the file to a buffer at once
  //   char *buffer = new char[length];
  //   binFile.read(buffer, length);
  //   binFile.close();

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
  // p_stocklevel
  // ((storage::ColumnStore *)this->table_part)
  //     ->load_data_from_binary(
  //         "p_stocklevel",
  //         concat_path(this->data_path, "part.csv.p_stocklevel").c_str());

  // uint64_t load_data_from_binary(std::string col_name, std::string
  // file_path);

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
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_orderpriority", storage::STRING, sizeof(tmp.lo_orderpriority)));
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_shippriority", storage::STRING, sizeof(tmp.lo_shippriority)));
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
  columns.emplace_back(std::tuple<std::string, storage::data_type, size_t>(
      "lo_shipmode", storage::STRING, sizeof(tmp.lo_shipmode)));

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
                                    num_parts);
}

MicroSSB::~MicroSSB() {}
MicroSSB::MicroSSB(std::string name, std::string binary_path_root)
    : Benchmark(name), data_path(binary_path_root) {
  std::cout << "Benchmark Init: " << name << std::endl;
  this->schema = &storage::Schema::getInstance();
  create_tbl_part();
  create_tbl_lineorder();
  // this->load_part();
  // this->load_lineorder();

  std::cout << "Total Memory Reserved for Tables: "
            << (double)this->schema->total_mem_reserved / (1024 * 1024 * 1024)
            << " GB" << std::endl;
  std::cout << "Total Memory Reserved for Deltas: "
            << (double)this->schema->total_delta_mem_reserved /
                   (1024 * 1024 * 1024)
            << " GB" << std::endl;
}

};  // namespace bench
