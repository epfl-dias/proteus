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

#ifndef GLO_HPP_
#define GLO_HPP_

#include <mutex>

#include "../indexes/hash_array.hpp"
#include "../indexes/hash_index.hpp"
#include "../indexes/index.hpp"
//#include "../indexes/ART/adaptive_radix_tree_index.hpp"
#include "snapshot/arena.hpp"
#include "snapshot/snapshot_manager.hpp"
#include "transactions/cc.hpp"

#define DEFAULT_OLAP_SOCKET 0
#define HTAP_ETL true  // for this, double master should be turned on too.

extern uint g_num_partitions;
extern uint g_delta_size;

#define __likely(x) __builtin_expect(x, 1)
#define __unlikely(x) __builtin_expect(x, 0)

namespace global_conf {

constexpr int MAX_PARTITIONS = 8;

using SnapshotManager = aeolus::snapshot::SnapshotManager;
using ConcurrencyControl = txn::CC_MV2PL;  // CC_GlobalLock;
using IndexVal = ConcurrencyControl::PRIMARY_INDEX_VAL;
const bool cc_ismv = ConcurrencyControl::is_mv();
using mv_version_list = txn::VERSION_LIST;
using mv_version = txn::VERSION;

template <typename T_KEY = uint64_t>
// using PrimaryIndex = indexes::HashIndex<T_KEY>;
using PrimaryIndex = indexes::HashArray<T_KEY>;
// using PrimaryIndex = indexes::AdaptiveRadixTreeIndex<T_KEY, void*>;

/* # of Snapshots*/
constexpr short num_master_versions = 2;
constexpr short num_delta_storages = 2;
constexpr bool reverse_partition_numa_mapping = false;

// for row-store inline bit, fix it to have bit-mask separate.
constexpr short HTAP_UPD_BIT_COUNT = 1;

}  // namespace global_conf

#endif /* GLO_HPP_ */
