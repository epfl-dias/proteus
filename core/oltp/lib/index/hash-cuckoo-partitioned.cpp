/*
     Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#include "oltp/index/hash-cuckoo-partitioned.hpp"

#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

#include "oltp/common/numa-partition-policy.hpp"

namespace indexes {
template <class K, class V>
CuckooPartitioned<K, V>::CuckooPartitioned(std::string name,
                                           size_t capacity_per_partition,
                                           uint64_t reserved_capacity)
    : HashIndex<K, V>(name),
      capacity(reserved_capacity),
      partitions(g_num_partitions) {
  this->capacity_per_partition = capacity_per_partition;
  this->capacity = capacity_per_partition * g_num_partitions;

  LOG(INFO) << "Creating a CuckooPartitioned[" << name
            << "] of size: " << reserved_capacity
            << " with partitions:" << (uint)partitions
            << " each with at least: " << capacity_per_partition;

  LOG_IF(FATAL, (capacity_per_partition * partitions) < reserved_capacity)
      << "Capacity per partition is less than the requested reserved capacity.";

  indexPartitions = new libcuckoo::cuckoohash_map<K, V>*[this->partitions];
  // new indexes::HashIndex<uint64_t, uint32_t>*[this->num_partitions];

  const auto& topo = topology::getInstance();
  const auto& nodes = topo.getCpuNumaNodes();

  for (auto i = 0; i < partitions; i++) {
    set_exec_location_on_scope d{nodes[i % nodes.size()]};
    indexPartitions[i] = new libcuckoo::cuckoohash_map<K, V>();
    indexPartitions[i]->reserve(capacity_per_partition);
  }
}

template <class K, class V>
CuckooPartitioned<K, V>::CuckooPartitioned(std::string name,
                                           uint64_t reserved_capacity)
    : HashIndex<K, V>(name),
      capacity(reserved_capacity),
      partitions(g_num_partitions) {
  this->capacity_per_partition = reserved_capacity;
  this->capacity = reserved_capacity * g_num_partitions;

  LOG_IF(FATAL, capacity < reserved_capacity)
      << "Capacity should be greater than or equal to reserved capacity";

  LOG(INFO) << "Creating a CuckooPartitioned[" << name
            << "] of size: " << reserved_capacity
            << " with partitions:" << (int)partitions
            << " each with at least: " << capacity_per_partition;

  indexPartitions = new libcuckoo::cuckoohash_map<K, V>*[this->partitions];
  // new indexes::HashIndex<uint64_t, uint32_t>*[this->num_partitions];

  const auto& topo = topology::getInstance();
  const auto& nodes = topo.getCpuNumaNodes();

  for (auto i = 0; i < partitions; i++) {
    set_exec_location_on_scope d{nodes[i % nodes.size()]};
    indexPartitions[i] = new libcuckoo::cuckoohash_map<K, V>();
    indexPartitions[i]->reserve(reserved_capacity);
  }
}

template <class K, class V>
CuckooPartitioned<K, V>::~CuckooPartitioned() {
  for (auto i = 0; i < partitions; i++) {
    delete indexPartitions[i];
  }
}

template class CuckooPartitioned<uint64_t, void*>;

}  // namespace indexes
