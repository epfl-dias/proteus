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

#include "glo.hpp"

DEFINE_uint64(num_olap_clients, 1, "Number of OLAP clients");
DEFINE_uint64(num_olap_repeat, 10,
              "Number of OLAP clients repeat query sequence");
DEFINE_uint64(num_oltp_clients, 0,
              "Number of OLTP clients (default: cpu-only: one-numa-socket, "
              "gpu-only: all cpus");
DEFINE_string(plan_json, "", "Plan to execute, takes priority over plan_dir");
DEFINE_string(plan_dir, "inputs/plans/cpu-ssb",
              "Directory with plans to be executed");
DEFINE_string(inputs_dir, "inputs/", "Data and catalog directory");
DEFINE_bool(run_oltp, true, "Run OLTP");
DEFINE_bool(run_olap, true, "Run OLAP");
DEFINE_uint64(oltp_elastic_threshold, 0, "elastic_oltp cores");
DEFINE_uint64(ch_scale_factor, 0, "CH-Bench scale factor");
DEFINE_bool(gpu_olap, false, "OLAP on GPU, OLTP on CPU");

DEFINE_string(htap_mode, "ISOLATED",
              "OLAP Scheduling Mode: 1) ISOLATED, 2) COLOC 3) HYBRID-ISOLATED "
              "4) HYBRID-COLOC 5) ADAPTIVE");

DEFINE_bool(per_query_snapshot, true, "Query-level data freshness");
DEFINE_int64(etl_interval_ms, -1, "max ETL interval");
