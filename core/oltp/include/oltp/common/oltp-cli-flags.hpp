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

DECLARE_uint32(num_workers);
DECLARE_uint32(num_partitions);
DECLARE_int32(num_iter_per_worker);
DECLARE_uint32(delta_size);
DECLARE_bool(layout_column_store);
DECLARE_uint32(worker_sched_mode);
DECLARE_uint32(report_stat_sec);
DECLARE_uint32(elastic_workload);
DECLARE_uint32(migrate_worker);
DECLARE_uint32(switch_master_sec);
DECLARE_bool(use_hyperthreads);

// DECLARE_bool(reverse_partition_numa_mapping);
