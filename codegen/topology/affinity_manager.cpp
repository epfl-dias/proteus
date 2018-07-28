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


exec_location::exec_location(int gpu): gpu_device(gpu){
    cpus = topology::getInstance().getGpuByIndex(gpu_device).local_cpu_set;
}

int numa_node_of_gpu(int device){
    return topology::getInstance().getGpuByIndex(device).local_cpu;
}


exec_location::exec_location(const topology::gpunode &gpu): gpu_device(gpu.id){
    cpus = gpu.local_cpu_set;
}