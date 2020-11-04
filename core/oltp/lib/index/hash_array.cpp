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

#include "oltp/index/hash_array.hpp"

#include "oltp/common/constants.hpp"
#include "oltp/common/numa-partition-policy.hpp"

namespace indexes {

template <class K, class V>
HashArray<K, V>::HashArray(std::string name, rowid_t reserved_capacity)
    : Index<K, V>(name, reserved_capacity),
      capacity(reserved_capacity),
      partitions(g_num_partitions) {
  capacity_per_partition =
      (reserved_capacity / partitions) + (reserved_capacity % partitions);
  capacity = capacity_per_partition * partitions;

  LOG(INFO) << "Creating a hashindex[" << name
            << "] of size: " << reserved_capacity
            << " with partitions:" << partitions
            << " each with: " << capacity_per_partition;

  arr = (char ***)malloc(sizeof(char *) * partitions);

  size_t size_per_part = capacity_per_partition * sizeof(char *);

  for (auto i = 0; i < partitions; i++) {
    arr[i] = (char **)MemoryManager::mallocPinnedOnNode(
        size_per_part, storage::NUMAPartitionPolicy::getInstance()
                           .getPartitionInfo(i)
                           .numa_idx);
    assert(arr[i] != nullptr);
    filler[i] = 0;
  }

  for (auto i = 0; i < partitions; i++) {
    auto *pt = (uint64_t *)arr[i];
    uint64_t warmup_max = size_per_part / sizeof(uint64_t);
#pragma clang loop vectorize(enable)
    for (uint64_t j = 0; j < warmup_max; j++) pt[j] = 0;
  }
}

template <class K, class V>
HashArray<K, V>::~HashArray() {
  for (auto i = 0; i < partitions; i++) {
    MemoryManager::freePinned(arr[i]);
  }
}

template class HashArray<uint64_t, void *>;

}  // namespace indexes
