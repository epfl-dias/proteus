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

// YCSB
DEFINE_double(ycsb_write_ratio, 0.5, "Writer to reader ratio");
DEFINE_double(ycsb_zipf_theta, 0.5, "YCSB - Zipfian theta");
DEFINE_uint32(ycsb_num_cols, 1, "YCSB - # Columns");
DEFINE_uint32(ycsb_num_ops_per_txn, 10, "YCSB - # ops / txn");
DEFINE_uint32(ycsb_num_records, 0, "YCSB - # records");
DEFINE_uint32(ycsb_num_col_upd, 0, "# of cols to upd per op");
DEFINE_uint32(ycsb_num_col_read, 0, "# of cols to read per op");
DEFINE_uint32(ycsb_num_col_read_offset, 0, "column offset for reader-columns");

// TPC-C
DEFINE_uint32(tpcc_num_wh, 0, "TPC-C - # of Warehouses ( 0 = one per worker");
DEFINE_uint32(ch_scale_factor, 0, "CH-Bench scale factor");
DEFINE_uint32(tpcc_dist_threshold, 0, "TPC-C - Distributed txn threshold");
DEFINE_string(tpcc_csv_dir, "/scratch/data/ch100w/raw",
              "CSV Dir for loading tpc-c data (bench-2)");

DEFINE_uint32(benchmark, 0, "Select benchmark (0=ycsb, 1=tpcc)");

DEFINE_uint64(runtime, 60, "Time to run for (sec)");

DEFINE_string(cdf_path, "", "Path for saving CDFs");
