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

#include "scheduler/affinity_manager.hpp"

#include "scheduler/topology.hpp"

namespace scheduler {

static thread_local cpu_set_t thread_core_affinity = cpu_set_t{1};
static thread_local uint32_t thread_cpu_numa_node_affinity = 0;

void AffinityManager::set(const cpunumanode *cpu) {
  // affinity_cpu_set::set(cpu, cpu.local_cpu_set);

  thread_core_affinity = cpu->local_cpu_set;
  thread_cpu_numa_node_affinity = cpu->id;

  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                         &(cpu->local_cpu_set));
}

void AffinityManager::set(const core *core) {
  thread_cpu_numa_node_affinity = core->local_cpu;
  CPU_ZERO(&thread_core_affinity);
  CPU_SET(core->id, &thread_core_affinity);

  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                         &(thread_core_affinity));
}
/*
void affinity::set(const cpunumanode &cpu) {
  affinity_cpu_set::set(cpu, cpu.local_cpu_set);
}

void affinity::set(const core &core) {
  thread_cpu_numa_node_affinity = core.local_cpu;
  CPU_ZERO(&thread_core_affinity);
  CPU_SET(core.id, &thread_core_affinity);

  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                         &(thread_core_affinity));
}

const cpunumanode &affinity::get() {
  const auto &topo = Topology::getInstance();
  return topo.getCpuNumaNodeById(thread_cpu_numa_node_affinity);
}

void affinity_cpu_set::set(const cpunumanode &cpu, cpu_set_t cores) {
  thread_core_affinity = cores;
  thread_cpu_numa_node_affinity = cpu.id;

  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cores);
}

cpu_set_t affinity_cpu_set::get() { return thread_core_affinity; }*/

}  // namespace scheduler
