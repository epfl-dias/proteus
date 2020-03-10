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
#include "snapshot/arena.hpp"
#include "snapshot/snapshot_manager.hpp"
#include "transactions/cc.hpp"

#define diascld33 false
#define diascld40 false
#define diascld48 true

#define icc148 false

#define DEFAULT_OLAP_SOCKET 0

#define HTAP_ETL true  // for this, double master should be turned on too.

#define MAX_NUM_PARTITIONS 2  // w.r.t. we have a 4-socket machine.

#if icc148
#define NUM_SOCKETS 2
#endif

#if diascld48
#define NUM_SOCKETS 2
#endif

#if diascld33
#define NUM_SOCKETS 4
#endif

#if diascld40
#define NUM_SOCKETS 2
#endif

extern uint g_num_partitions;
extern uint g_delta_size;

#define __likely(x) __builtin_expect(x, 1)
#define __unlikely(x) __builtin_expect(x, 0)

namespace global_conf {

using SnapshotManager = aeolus::snapshot::SnapshotManager;

using ConcurrencyControl = txn::CC_MV2PL;  // CC_GlobalLock;
using IndexVal = ConcurrencyControl::PRIMARY_INDEX_VAL;

const bool cc_ismv = ConcurrencyControl::is_mv();

using mv_version_list = txn::VERSION_LIST;
using mv_version = txn::VERSION;

template <class T_KEY>
// using PrimaryIndex = indexes::HashIndex<T_KEY>;
using PrimaryIndex = indexes::HashArray<T_KEY>;

// const ushort NUM_SOCKETS =
//     scheduler::Topology::getInstance().getCpuNumaNodeCount();
// const ushort NUM_CORE_PER_SOCKET = scheduler::Topology::getInstance()
//                                        .getCpuNumaNodes()
//                                        .front()
//                                        .local_cores.size();
// const ushort MAX_WORKERS = scheduler::Topology::getInstance().getCoreCount();

// const uint time_master_switch_ms = 200;

/* # of Snapshots*/
constexpr short num_master_versions = 2;
constexpr short num_delta_storages = 2;
constexpr bool reverse_partition_numa_mapping = true;

// for row-store inline bit, fix it to have bit-mask separate.
constexpr short HTAP_UPD_BIT_COUNT = 1;

}  // namespace global_conf

#endif /* GLO_HPP_ */
