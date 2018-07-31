/*
    RAW -- High-performance querying over raw, never-seen-before data.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Labaratory (DIAS)
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


static thread_local uint32_t thread_cpu_numa_node_affinity = 0;

exec_location::exec_location(int gpu):
    exec_location(topology::getInstance().getGpuByIndex(gpu)){
}

int numa_node_of_gpu(int device){
    return topology::getInstance().getGpuByIndex(device).local_cpu;
}

exec_location::exec_location(const topology::gpunode &gpu):
    gpu_device(gpu.id),
    cpu(topology::getInstance().getCpuNumaNodeById(gpu.local_cpu)){
}

void set_affinity(const topology::cpunumanode &cpu){
    thread_cpu_numa_node_affinity = cpu.id;

#ifndef NDEBUG
    int err =
#endif
    pthread_setaffinity_np(pthread_self(),
                            sizeof(cpu_set_t),
                            &(cpu.local_cpu_set));
    assert(!err);
}

const topology::cpunumanode &get_affinity(){
    const auto &topo = topology::getInstance();
    return topo.getCpuNumaNodeById(thread_cpu_numa_node_affinity);
}

// inline set_device_on_scope(const topology::gpunode &gpu):
//         device(topology::getInstance().getActiveGpu()){
//     gpu_run(cudaSetDevice(gpu.getId()));
// }

// inline ~set_device_on_scope(){
//     gpu_run(cudaSetDevice(device.getId()));
// }