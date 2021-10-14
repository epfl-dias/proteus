/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#ifndef BENCH_TPCC_CONSTANTS_HPP
#define BENCH_TPCC_CONSTANTS_HPP

/*
  Benchmark: TPC-C
  Spec: http://www.tpc.org/tpc_documents_current_versions/pdf/tpc-c_v5.11.0.pdf
*/

#define PARTITION_LOCAL_ITEM_TABLE true
#define tpcc_dist_txns false
#define tpcc_cust_sec_idx false
#define batch_insert_no_ol true

// Space constraint for benchmarks.
#define index_on_order_tbl false  // also cascade to orderline, neworder table
#define payment_txn_insert_history false

//#if diascld40
//#define TPCC_MAX_ORD_PER_DIST 200000  // 2 master - 2 socket
//#elif diascld48
//#define TPCC_MAX_ORD_PER_DIST 350000  // 2 master - 2 socket
//#elif icc148
//#define TPCC_MAX_ORD_PER_DIST 2000000  // 2 master - 1 socket
//#else
//#define TPCC_MAX_ORD_PER_DIST 200000
//#endif

// ic149
//#define TPCC_MAX_ORD_PER_DIST 1000000
// dias33
#define TPCC_MAX_ORD_PER_DIST 150000

// Payment + NewOrder Only
#define NO_MIX 51
#define P_MIX 49
#define OS_MIX 0
#define D_MIX 0
#define SL_MIX 0
#define MIX_COUNT 100

// NewOrder Only
/*
#define NO_MIX 100
#define P_MIX 0
#define OS_MIX 0
#define D_MIX 0
#define SL_MIX 0
#define MIX_COUNT 100
*/

// Standard TPC-C Mix
/*
#define NO_MIX 45
#define P_MIX 43
#define OS_MIX 4
#define D_MIX 4
#define SL_MIX 4
#define MIX_COUNT 100
*/

#define FIRST_NAME_MIN_LEN 8
#define FIRST_NAME_LEN 16
#define LAST_NAME_LEN 16
#define TPCC_MAX_OL_PER_ORDER 15
#define MAX_OPS_PER_QUERY 255

// From TPCC-SPEC
#define TPCC_MAX_ITEMS 100000
#define TPCC_NCUST_PER_DIST 3000
#define TPCC_NDIST_PER_WH 10
#define TPCC_ORD_PER_DIST 3000

// TPC-C Specific PK generation for composite PK.
#define MAKE_STOCK_KEY(w, s) ((w)*TPCC_MAX_ITEMS + (s))
#define MAKE_DIST_KEY(w, d) ((w)*TPCC_NDIST_PER_WH + (d))
#define MAKE_CUST_KEY(w, d, c) (MAKE_DIST_KEY(w, d) * TPCC_NCUST_PER_DIST + (c))

#define MAKE_ORDER_KEY(w, d, o) \
  ((MAKE_DIST_KEY(w, d) * TPCC_MAX_ORD_PER_DIST) + (o))
#define MAKE_OL_KEY(w, d, o, ol) \
  (MAKE_ORDER_KEY(w, d, o) * TPCC_MAX_OL_PER_ORDER + (ol))

// Constants and TypeDefs for TPC-C
using date_t = uint64_t;
constexpr char csv_delim = '|';

#endif  // BENCH_TPCC_CONSTANTS_HPP
