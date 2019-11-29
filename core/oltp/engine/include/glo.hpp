/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                             Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef GLO_HPP_
#define GLO_HPP_

#include <mutex>

#include "indexes/hash_array.hpp"
#include "indexes/hash_index.hpp"
#include "scheduler/topology.hpp"
#include "snapshot/arena.hpp"
#include "snapshot/snapshot_manager.hpp"
#include "transactions/cc.hpp"

#define diascld33 true
#define diascld40 false
#define diascld48 false

#define icc148 false

#define DEFAULT_OLAP_SOCKET 0

#define HTAP_DOUBLE_MASTER true
#define HTAP_COW false
#define HTAP_ETL false  // for this, double master should be turned on too.

// Memory Allocators
#define HTAP_RM_SERVER false
#define PROTEUS_MEM_MANAGER false
#define SHARED_MEMORY false  // if htap=false, then shm or numa_alloc

#if icc148
#define NUM_SOCKETS 2
#define NUM_CORE_PER_SOCKET 28
#endif

#if diascld48
#define NUM_SOCKETS 2
#define NUM_CORE_PER_SOCKET 24
#endif

#if diascld33
#define NUM_SOCKETS 4
#define NUM_CORE_PER_SOCKET 18
//#define MAX_WORKERS 72
#endif

#if diascld40
#define NUM_SOCKETS 2
#define NUM_CORE_PER_SOCKET 64
//#define MAX_WORKERS 128
#endif

#if HTAP_DOUBLE_MASTER
#define HTAP_UPD_BIT_COUNT 1
#else
#define HTAP_UPD_BIT_COUNT 0
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
constexpr short num_master_versions = 1;
constexpr short num_delta_storages = 2;

constexpr bool reverse_partition_numa_mapping = false;

}  // namespace global_conf

#endif /* GLO_HPP_ */
