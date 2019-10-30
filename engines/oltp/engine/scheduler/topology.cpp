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

#include "scheduler/topology.hpp"

#include <unistd.h>

#include <cmath>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <vector>

//#include "topology/affinity_manager.hpp"

#include <numa.h>
#include <numaif.h>

namespace scheduler {

// template<typename T>
/*
void *topology::cpunumanode::alloc(size_t bytes) const {
  return numa_alloc_onnode(bytes, id);
}

void topology::cpunumanode::free(void *mem, size_t bytes) {
  numa_free(mem, bytes);
}*/

Topology::Topology() {
  // Creating gpunodes requires that we know the number of cores,
  // so start by reading the CPU configuration
  core_cnt = sysconf(_SC_NPROCESSORS_ONLN);
  assert(core_cnt > 0);

  std::map<uint32_t, std::vector<uint32_t>> numa_to_cores_mapping;

  for (uint32_t j = 0; j < core_cnt; ++j) {
    numa_to_cores_mapping[numa_node_of_cpu(j)].emplace_back(j);
  }

  uint32_t max_numa_id = 0;
  for (const auto &numa : numa_to_cores_mapping) {
    cpu_info.emplace_back(numa.first, numa.second, cpu_info.size());
    max_numa_id = std::max(max_numa_id, cpu_info.back().id);
  }

  for (auto &n : cpu_info) {
    for (const auto &n2 : cpu_info) {
      n.distance.emplace_back(numa_distance(n.id, n2.id));
    }
  }

  cpunuma_index.resize(max_numa_id + 1);
  for (auto &ci : cpunuma_index) ci = 0;

  for (size_t i = 0; i < cpu_info.size(); ++i) {
    cpunuma_index[cpu_info[i].id] = i;
  }

  for (const auto &cpu : cpu_info) {
    for (const auto &core : cpu.local_cores) {
      core_info.emplace_back(core, cpu.id, core_info.size(), cpu.index_in_topo);
    }
  }

  assert(core_info.size() == core_cnt);
}

std::ostream &operator<<(std::ostream &out, const Topology &topo) {
  out << "numa nodes: " << topo.getCpuNumaNodeCount() << "\n";
  out << "core count: " << topo.getCoreCount() << "\n";

  out << '\n';

  char core_mask[topo.core_cnt + 1];
  core_mask[topo.core_cnt] = '\0';

  uint32_t digits = (uint32_t)std::ceil(std::log10(topo.core_cnt));

  for (uint32_t k = digits; k > 0; --k) {
    uint32_t base = std::pow(10, k - 1);

    if (k == ((digits + 1) / 2))
      out << "core: ";
    else
      out << "      ";

    if (1 == digits)
      out << ' ';
    else if (k == digits)
      out << '/';
    else if (k == 1)
      out << '\\';
    else
      out << '|';
    out << std::setw(base + 4 + 4 + 3 + 18) << ((k == 1) ? '0' : ' ');

    for (uint32_t i = base; i < topo.core_cnt; ++i) {
      out << (i / base) % 10;
    }
    out << '\n';
  }

  for (const auto &node : topo.getCpuNumaNodes()) {
    out << "node: " << std::setw(6) << node.id << " | ";

    out << std::setw(4 + 4 + 3) << ' ' << " | ";

    out << "cores: ";

    // for ( auto cpu_id : node.logical_cpus) {
    //     out << std::setw(4) << cpu_id << " ";
    // }

    for (uint32_t i = 0; i < topo.core_cnt; ++i) core_mask[i] = ' ';

    for (auto cpu_id : node.local_cores) core_mask[cpu_id] = 'x';

    out << core_mask << '\n';
  }

  out << '\n';

  // size_t sockets = topo.cpu_info.size();

  out << '\n';

  for (const auto &node : topo.getCpuNumaNodes()) {
    out << "node: ";
    out << node.id << " | ";
    for (auto d : node.distance) out << std::setw(4) << d;
    out << '\n';
  }

  out << '\n';
  return out;
}

cpunumanode::cpunumanode(uint32_t id, const std::vector<uint32_t> &local_cores,
                         uint32_t index_in_topo)
    : id(id),
      // distance(b.distance),
      index_in_topo(index_in_topo),
      local_cores(local_cores) {
  CPU_ZERO(&local_cpu_set);
  for (const auto &c : local_cores) CPU_SET(c, &local_cpu_set);
}

// Topology Topology::instance;
}  // namespace scheduler
