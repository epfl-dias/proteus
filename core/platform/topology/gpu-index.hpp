/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef PROTEUS_GPU_INDEX_HPP
#define PROTEUS_GPU_INDEX_HPP

#include <vector>

#include "topology.hpp"

class gpu_index {
 public:
  std::vector<size_t> d;

 public:
  gpu_index() {
    const auto &topo = topology::getInstance();
    d.reserve(topo.getGpuCount());
    size_t cpus = topo.getCpuNumaNodeCount();
    for (size_t j = 0; d.size() < topo.getGpuCount(); ++j) {
      size_t cpu = j % cpus;
      size_t gpur = j / cpus;
      const auto &numanode = topo.getCpuNumaNodes()[cpu];
      const auto &gpus = numanode.local_gpus;
      if (gpur >= gpus.size()) continue;
      d.emplace_back(gpus[gpur]);
    }
  }
};

#endif  // PROTEUS_GPU_INDEX_HPP
