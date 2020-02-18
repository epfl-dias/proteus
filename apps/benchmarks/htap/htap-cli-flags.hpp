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

DECLARE_uint64(num_olap_clients);
DECLARE_uint64(num_olap_repeat);
DECLARE_uint64(num_oltp_clients);
DECLARE_string(plan_json);
DECLARE_string(plan_dir);
DECLARE_string(inputs_dir);
DECLARE_bool(run_oltp);
DECLARE_bool(run_olap);
DECLARE_uint64(oltp_elastic_threshold);
DECLARE_uint64(oltp_coloc_threshold);
DECLARE_uint64(ch_scale_factor);

DECLARE_bool(gpu_olap);
DECLARE_string(htap_mode);
DECLARE_bool(per_query_snapshot);
DECLARE_int64(etl_interval_ms);

DECLARE_int64(micro_ch_query);
DECLARE_double(adaptive_ratio);
