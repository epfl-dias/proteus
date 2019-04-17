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

#ifndef BENCH_HPP_
#define BENCH_HPP_

#include <iostream>
#include <string>

namespace bench {

// enum OP_TYPE { OPTYPE_LOOKUP, OPTYPE_UPDATE }; done in txn_manager

class Benchmark {
 public:
  virtual void init() {
  }  // who will init the bench? the main session or in worker's init?
  virtual void load_data(int num_threads = 1) {}
  virtual void *gen_txn(int wid, void *txn_ptr) { return nullptr; }
  virtual bool exec_txn(void *stmts, uint64_t xid, ushort master_ver,
                        ushort delta_ver) {
    return true;
  }
  virtual void *get_query_struct_ptr() { return nullptr; }

  std::string name;

  // private:
  Benchmark(std::string name = "DUMMY") : name(name){};

  virtual ~Benchmark() { std::cout << "destructor of Benchmark" << std::endl; }
};

}  // namespace bench

#endif /* BENCH_HPP_ */
