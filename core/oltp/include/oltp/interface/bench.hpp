/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef BENCH_HPP_
#define BENCH_HPP_

#include <iostream>
#include <string>

namespace bench {

class Benchmark {
 public:
  const std::string name;
  ushort num_active_workers;
  const ushort num_max_workers;
  const ushort num_partitions;

  virtual void init() {}
  virtual void deinit() {}
  virtual void load_data(int num_threads = 1) {}
  virtual void gen_txn(int wid, void *txn_ptr, ushort partition_id) {}
  virtual bool exec_txn(const void *stmts, uint64_t xid, ushort master_ver,
                        ushort delta_ver, ushort partition_id) {
    return true;
  }

  // Should return a memory pointer which will be used to describe a query.
  virtual void *get_query_struct_ptr(ushort pid) { return nullptr; }
  virtual void free_query_struct_ptr(void *ptr) {}

  // NOTE: Following will run before/after the workers starts the execution. it
  // will be synchronized, i.e., worker will not start transaction until all
  // workers finish the pre-run and a worker will not start post-run unless all
  // workers are ready to start the post run. Moreover, this will not apply to
  // the hot-plugged workers.
  virtual void post_run(int wid, uint64_t xid, ushort partition_id,
                        ushort master_ver) {}
  virtual void pre_run(int wid, uint64_t xid, ushort partition_id,
                       ushort master_ver) {}

  Benchmark(std::string name = "BENNCH-DUMMY", ushort num_active_workers = 1,
            ushort num_max_workers = 1, ushort num_partitions = 1)
      : name(name),
        num_active_workers(num_active_workers),
        num_max_workers(num_max_workers),
        num_partitions(num_partitions) {}

  virtual ~Benchmark() {}
};

}  // namespace bench

#endif /* BENCH_HPP_ */
