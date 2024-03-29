/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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
#include <gflags/gflags.h>

#include <iostream>
#include <oltp/common/oltp-cli-flags.hpp>

DEFINE_uint32(num_workers, 0, "Number of txn-workers");
DEFINE_int32(num_iter_per_worker, -1, "Iterations per worker");
DEFINE_uint32(num_partitions, 0,
              "Number of storage partitions (round robin NUMA nodes)");
DEFINE_uint32(delta_size, 2, "Size of delta storage in GBs.");
DEFINE_uint32(report_stat_sec, 0, "Report stats every x secs");
DEFINE_uint32(elastic_workload, 0, "if > 0, add a worker every x seconds");
DEFINE_uint32(migrate_worker, 0, "if > 0, migrate worker to other side");
DEFINE_uint32(switch_master_sec, 0, "if > 0, add a worker every x seconds");

// Instead of the following, take schedulingPolicy as input, which will dictate
// the use of hyper-threads implicitly
DEFINE_bool(use_hyperthreads, false, "OLTP workers to use hyper threads.");

// DEFINE_bool(reverse_partition_numa_mapping, true,
//             "True: lowest PID mapped to the highest NUMA ID");
