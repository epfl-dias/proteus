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

#include "topology/affinity_manager.hpp"

#include "topology/topology.hpp"

static thread_local cpu_set_t thread_core_affinity = cpu_set_t{1};
static thread_local uint32_t thread_cpu_numa_node_affinity = 0;
static thread_local uint32_t thread_server_affinity = 0;

exec_location::exec_location(int gpu)
    : exec_location(topology::getInstance().getGpuByIndex(gpu)) {}

int numa_node_of_gpu(int device) {
  return topology::getInstance().getGpuByIndex(device).local_cpu;
}

exec_location::exec_location(const topology::gpunode &gpu)
    : gpu_device(gpu.id),
      cpu(topology::getInstance().getCpuNumaNodeById(gpu.local_cpu)),
      cores(topology::getInstance()
                .getCpuNumaNodeById(gpu.local_cpu)
                .local_cpu_set) {}

void affinity::set(const topology::cpunumanode &cpu) {
  affinity_cpu_set::set(cpu, cpu.local_cpu_set);
}

void affinity::set(const topology::core &core) {
  thread_cpu_numa_node_affinity = core.local_cpu;
  CPU_ZERO(&thread_core_affinity);
  CPU_SET(core.id, &thread_core_affinity);

#ifndef NDEBUG
  int err =
#endif
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t),
                             &(thread_core_affinity));
  assert(!err);
}

void affinity::set_server(int32_t server) { thread_server_affinity = server; }
int32_t affinity::get_server() { return thread_server_affinity; }

const topology::cpunumanode &affinity::get() {
  const auto &topo = topology::getInstance();
  return topo.getCpuNumaNodeById(thread_cpu_numa_node_affinity);
}

void affinity_cpu_set::set(const topology::cpunumanode &cpu, cpu_set_t cores) {
  thread_core_affinity = cores;
  thread_cpu_numa_node_affinity = cpu.id;

#ifndef NDEBUG
  int err =
#endif
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cores);
  assert(!err);
}

cpu_set_t affinity_cpu_set::get() { return thread_core_affinity; }

// inline set_device_on_scope(const topology::gpunode &gpu):
//         device(topology::getInstance().getActiveGpu()){
//     gpu_run(cudaSetDevice(gpu.getId()));
// }

// inline ~set_device_on_scope(){
//     gpu_run(cudaSetDevice(device.getId()));
// }
