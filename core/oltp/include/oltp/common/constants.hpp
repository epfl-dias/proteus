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

#ifndef PROTEUS_OLTP_CONSTANTS_HPP
#define PROTEUS_OLTP_CONSTANTS_HPP

#define HTAP_ETL false  // for this, double master should be turned on too.

#include "oltp/common/common.hpp"
#include "oltp/index/hash_array.hpp"
#include "oltp/index/hash_index.hpp"
#include "oltp/snapshot/snapshot_manager.hpp"
#include "oltp/transaction/concurrency-control/concurrency-control.hpp"

namespace global_conf {

using SnapshotManager = aeolus::snapshot::SnapshotManager;
using ConcurrencyControl = txn::CC_MV2PL;
using IndexVal = ConcurrencyControl::PRIMARY_INDEX_VAL;

template <typename T_KEY = uint64_t>
// using PrimaryIndex = indexes::HashArray<T_KEY>;
using PrimaryIndex = indexes::HashIndex<T_KEY>;
// using PrimaryIndex = indexes::AdaptiveRadixTreeIndex<T_KEY, void*>;

constexpr int MAX_PARTITIONS = 8;
constexpr master_version_t num_master_versions = 1;
constexpr delta_id_t num_delta_storages = 2;
constexpr bool reverse_partition_numa_mapping = false;
constexpr uint DEFAULT_OLAP_SOCKET = 0;

static_assert((!HTAP_ETL) || (HTAP_ETL && num_master_versions >= 2),
              "For ETL-based HTAP, # of master-versions should be >= 2");

static_assert(!(GcMechanism == GcTypes::OneShot) ||
                  ((GcMechanism == GcTypes::OneShot) && num_delta_storages > 1),
              "OneShot GC requires number of delta storages > 1");

}  // namespace global_conf

#endif /* PROTEUS_OLTP_CONSTANTS_HPP */
