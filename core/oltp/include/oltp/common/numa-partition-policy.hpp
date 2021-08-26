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

#ifndef PROTEUS_NUMA_PARTITION_POLICY_HPP
#define PROTEUS_NUMA_PARTITION_POLICY_HPP

#include <cassert>
#include <iostream>
#include <vector>

namespace storage {

class NUMAPartitionPolicy {
 public:
  // Singleton
  static inline NUMAPartitionPolicy &getInstance() {
    static NUMAPartitionPolicy instance;
    return instance;
  }
  NUMAPartitionPolicy(NUMAPartitionPolicy const &) = delete;  // Don't Implement
  void operator=(NUMAPartitionPolicy const &) = delete;       // Don't implement

  class TablePartition {
   public:
    const int pid;
    const int numa_idx;

   public:
    explicit TablePartition(int pid, int numa_idx)
        : pid(pid), numa_idx(numa_idx) {}

    inline bool operator==(const TablePartition &o) const {
      return (pid == o.pid && numa_idx == o.numa_idx);
    }
    friend std::ostream &operator<<(std::ostream &out, const TablePartition &r);
  };

  const TablePartition &getPartitionInfo(uint pid) {
    assert(pid < PartitionVector.size());
    return PartitionVector[pid];
  }

  [[maybe_unused]] auto getPartitions() { return PartitionVector; }

  uint getDefaultPartition() {
    assert(!PartitionVector.empty());
    return PartitionVector[0].numa_idx;
  }

 private:
  std::vector<TablePartition> PartitionVector{};

  NUMAPartitionPolicy();

  friend std::ostream &operator<<(std::ostream &out,
                                  const NUMAPartitionPolicy &r);
};

}  // namespace storage

#endif  // PROTEUS_NUMA_PARTITION_POLICY_HPP
