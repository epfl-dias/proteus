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

#include <gflags/gflags.h>
#include <iostream>
#include <string>

DECLARE_uint64(num_partitions);

namespace bench {

// enum OP_TYPE { OPTYPE_LOOKUP, OPTYPE_UPDATE }; done in txn_manager

class Benchmark {
 public:
  virtual void init() {
  }  // who will init the bench? the main session or in worker's init?
  virtual void load_data(int num_threads = 1) {}
  virtual void gen_txn(int wid, void *txn_ptr, ushort partition_id) {}
  virtual bool exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                        ushort delta_ver, ushort partition_id) {
    return true;
  }

  // Should return a memory pointer which will be used to describe a query.
  virtual void *get_query_struct_ptr(ushort pid = 0) { return nullptr; }
  virtual void free_query_struct_ptr(void *ptr) {}

  // NOTE: Following will run before/after the workers starts the execution. it
  // will be synchorinized, i.e., worker will not start transaction until all
  // workers finish the pre-run and a worker will not start post-run unless all
  // workers are ready to start the post run. Morever, this will not apply to
  // the hotplugged workers.

  // TODO: not implemented in the worker pool as of yet.

  virtual void post_run(int wid, uint64_t xid, ushort partition_id,
                        ushort master_ver) {}
  virtual void pre_run(int wid, uint64_t xid, ushort partition_id,
                       ushort master_ver) {}

  std::string name;

  // private:
  Benchmark(std::string name = "DUMMY") : name(name) {}

  virtual ~Benchmark() { std::cout << "destructor of Benchmark" << std::endl; }
};

}  // namespace bench

#endif /* BENCH_HPP_ */
