/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#ifndef PROTEUS_OLTP_COMMON_HPP
#define PROTEUS_OLTP_COMMON_HPP

#define INSTRUMENTATION false

#include <map>
#include <platform/common/common.hpp>
#include <platform/util/timing.hpp>

#define __likely(x) __builtin_expect(x, 1)
#define __unlikely(x) __builtin_expect(x, 0)

// DO NOT CHANGE!
typedef uint64_t rowid_t;
typedef uint8_t delta_id_t;
typedef uint64_t xid_t;
typedef uint8_t worker_id_t;
typedef uint8_t table_id_t;
typedef uint8_t column_id_t;
typedef uint8_t partition_id_t;

constexpr auto MAX_N_COLUMNS = UINT8_MAX;

// TODO: replace column-widths with
// typedef uint16_t column_width_t;

// Circular-Master
typedef uint8_t master_version_t;

// Lazy-Master
typedef int8_t snapshot_version_t;

typedef std::map<uint64_t, std::string> dict_dstring_t;

extern uint g_num_partitions;
extern uint g_delta_size;

enum SnapshotTypes { None, CircularMaster, LazyMaster, MVCC };

constexpr auto DefaultSnapshotMechanism = SnapshotTypes::LazyMaster;

constexpr int MAX_PARTITIONS = 8;

#endif  // PROTEUS_OLTP_COMMON_HPP
