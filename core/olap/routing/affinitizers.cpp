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

#include "routing/routing-policy.hpp"
#include "topology/topology.hpp"

AffinityPolicy::AffinityPolicy(size_t fanout, const Affinitizer *aff)
    : aff(aff) {
  indexes.resize(aff->size());
  for (size_t i = 0; i < fanout; ++i) {
    indexes[aff->getAvailableCUIndex(i)].emplace_back(i);
  }
  for (auto &ind : indexes) {
    if (!ind.empty()) continue;
    for (size_t i = 0; i < fanout; ++i) ind.emplace_back(i);
  }
}

size_t AffinityPolicy::getIndexOfRandLocalCU(void *p) const {
  auto r = rand();

  auto index_in_topo = aff->getLocalCUIndex(p);

  const auto &ind = indexes[index_in_topo];
  return ind[r % ind.size()];
}

std::unique_ptr<Affinitizer> getDefaultAffinitizer(DeviceType d) {
  switch (d) {
    case DeviceType::CPU:
      return std::make_unique<CpuNumaNodeAffinitizer>();
    case DeviceType::GPU:
      return std::make_unique<GPUAffinitizer>();
  }
}
