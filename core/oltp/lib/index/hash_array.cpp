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
#include "oltp/storage/table.hpp"

namespace indexes {

template <class K, class V>
HashArray<K, V>::HashArray(std::string name, uint64_t num_obj)
    : capacity(num_obj), name(name) {
  uint FLAGS_num_partitions = g_num_partitions;

  this->partitions = FLAGS_num_partitions;

  capacity_per_partition = (num_obj / partitions) + (num_obj % partitions);
  capacity = capacity_per_partition * partitions;

  std::cout << "Creating a hashindex[" << name << "] of size: " << num_obj
            << " with partitions:" << FLAGS_num_partitions
            << " each with: " << capacity_per_partition << std::endl;

  arr = (char ***)malloc(sizeof(char *) * partitions);

  size_t size_per_part = capacity_per_partition * sizeof(char *);

  for (int i = 0; i < partitions; i++) {
    arr[i] = (char **)MemoryManager::mallocPinnedOnNode(
        size_per_part, storage::NUMAPartitionPolicy::getInstance()
                           .getPartitionInfo(i)
                           .numa_idx);
    assert(arr[i] != nullptr);
    filler[i] = 0;
  }

  for (int i = 0; i < partitions; i++) {
    uint64_t *pt = (uint64_t *)arr[i];
    uint64_t warmup_max = size_per_part / sizeof(uint64_t);
#pragma clang loop vectorize(enable)
    for (uint64_t j = 0; j < warmup_max; j++) pt[j] = 0;
  }
}

template <class K, class V>
HashArray<K, V>::~HashArray() {
  for (int i = 0; i < partitions; i++) {
    MemoryManager::freePinned(arr[i]);
  }
}

template class HashArray<uint64_t, void *>;

}  // namespace indexes
