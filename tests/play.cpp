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

#include <unistd.h>

#include <functional>
#include <iostream>
#include <tuple>

#include "benchmarks/ycsb.hpp"
#include "indexes/hash_index.hpp"
#include "scheduler/topology.hpp"
#include "scheduler/worker.hpp"
#include "storage/memory_manager.hpp"
#include "storage/table.hpp"
#include "transactions/transaction_manager.hpp"
#include "utils/utils.hpp"

int main() {
  /* write all the shit and testing code here */
  std::cout << "play ground" << std::endl;
}

void ycsb_gen_txn() {
  bench::YCSB* ycsb_bench = new bench::YCSB();
  // ycsb_bench->load_data();

  std::cout << "------------------------------------" << std::endl;
  std::cout << "YCSB gen query test" << std::endl;

  for (int i = 0; i < 24; i++) {
    std::cout << "--------WORKER " << i << "----" << std::endl;
    for (int j = 0; j < 10; j++) {
      std::cout << *((txn::TXN*)ycsb_bench->gen_txn(i));
    }
  }

  std::cout << "------------------------------------" << std::endl;
}

void hash_test() {
  /* ------------------------------------ */
  /* HASH INDEX TEST*/
  /*
  indexes::HashIndex<uint64_t, std::string> cc;

  for (int i = 0; i < 100; i++) {
    cc.insert(i, "hello");
  }

  for (int i = 0; i < 101; i++) {
    std::string out;

    if (cc.find(i, out)) {
      std::cout << i << "  " << out << std::endl;
    } else {
      std::cout << i << "  NOT FOUND" << std::endl;
    }
  }
  return 0;
  */
}
