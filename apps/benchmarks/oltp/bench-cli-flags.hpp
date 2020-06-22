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

// YCSB
DECLARE_double(ycsb_write_ratio);
DECLARE_double(ycsb_zipf_theta);
DECLARE_uint32(ycsb_num_cols);
DECLARE_uint32(ycsb_num_ops_per_txn);
DECLARE_uint32(ycsb_num_records);
DECLARE_uint32(ycsb_num_col_upd);
DECLARE_uint32(ycsb_num_col_read);

// TPC-C
DECLARE_uint32(tpcc_num_wh);
DECLARE_uint32(ch_scale_factor);
DECLARE_uint32(tpcc_dist_threshold);
DECLARE_string(tpcc_csv_dir);

DECLARE_uint32(benchmark);
DECLARE_uint64(runtime);
