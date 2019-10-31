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


DEFINE_uint64(num_workers, 0, "Number of txn-workers");
DEFINE_uint64(num_partitions, 0,
              "Number of storage partitions ( round robin NUMA nodes)");
DEFINE_uint64(delta_size, 8, "Size of delta storage in GBs.");
DEFINE_bool(layout_column_store, true, "True: ColumnStore / False: RowStore");
DEFINE_uint64(worker_sched_mode, 0,
              "Scheduling of worker: 0-default, 1-interleaved-even, "
              "2-interleaved-odd, 3-reversed.");
DEFINE_uint64(report_stat_sec, 0, "Report stats every x secs");
DEFINE_uint64(elastic_workload, 0, "if > 0, add a worker every x seconds");
DEFINE_uint64(migrate_worker, 0, "if > 0, migrate worker to other side");
DEFINE_uint64(switch_master_sec, 0, "if > 0, add a worker every x seconds");
