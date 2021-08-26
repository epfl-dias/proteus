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

#include "oltp/common/numa-partition-policy.hpp"

#include "oltp/common/common.hpp"
#include "oltp/common/constants.hpp"

namespace storage {

NUMAPartitionPolicy::NUMAPartitionPolicy() {
  auto num_numa_nodes = topology::getInstance().getCpuNumaNodeCount();
  for (int i = 0; i < g_num_partitions; i++) {
    if (global_conf::reverse_partition_numa_mapping)
      PartitionVector.emplace_back(
          TablePartition{i, static_cast<int>(num_numa_nodes - i - 1)});
    else
      PartitionVector.emplace_back(TablePartition{i, i});
  }
  LOG(INFO) << *this;
}

std::ostream& operator<<(std::ostream& out,
                         const NUMAPartitionPolicy::TablePartition& r) {
  out << "\tPID: " << r.pid << "|\t NUMA Index: " << r.numa_idx << std::endl;
  return out;
}

std::ostream& operator<<(std::ostream& out, const NUMAPartitionPolicy& r) {
  out << "\n---------------------------" << std::endl;
  out << "NUMA Partition Policy" << std::endl;
  out << "\treverse_numa_mapping: "
      << (global_conf::reverse_partition_numa_mapping ? "True" : "False")
      << std::endl;
  out << "---------------------------" << std::endl;
  out << "Default Partition:" << std::endl;
  out << r.PartitionVector[0];
  out << "---------------------------" << std::endl;
  for (const auto& p : r.PartitionVector) {
    out << p;
  }
  out << "---------------------------" << std::endl;
  return out;
}

}  // namespace storage
