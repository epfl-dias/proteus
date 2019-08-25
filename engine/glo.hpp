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

#include "indexes/hash_array.hpp"
#include "indexes/hash_index.hpp"
#include "transactions/cc.hpp"

#include "snapshot/arena.hpp"
#include "snapshot/snapshot_manager.hpp"

#define NUM_SOCKETS 4
#define NUM_CORE_PER_SOCKET 36
#define MAX_WORKERS 144
#define DELTA_SIZE 2  // 2G // 6442450944 6G
#define HTAP_RM_SERVER false
#define HTAP_DOUBLE_MASTER true
#define HTAP_COW false
#define HTAP_UPD_BIT_MASK false
#define SHARED_MEMORY false  // if htap=false, then shm or numa_alloc

// typedef cuckoohash_map<std::string, std::string> HashIndex;

// template <class hash_val, class key = uint64_t>
// using HashIndex = cuckoohash_map<key, hash_val>;

namespace global_conf {

using SnapshotManager = aeolus::snapshot::SnapshotManager;

using ConcurrencyControl = txn::CC_MV2PL;  // CC_GlobalLock;
using IndexVal = ConcurrencyControl::PRIMARY_INDEX_VAL;

const bool cc_ismv = ConcurrencyControl::is_mv();

using mv_version_list = txn::VERSION_LIST;
using mv_version = txn::VERSION;

template <class T_KEY>
using PrimaryIndex = indexes::HashIndex<T_KEY>;
// using PrimaryIndex = indexes::HashArray<T_KEY>;

// const uint time_master_switch_ms = 200;

/* # of Snapshots*/
const short num_master_versions = 2;
const short num_delta_storages = 2;

const int master_col_numa_id = 0;

}  // namespace global_conf

#endif /* GLO_HPP_ */
