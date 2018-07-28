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

#ifndef TOPOLOGY_HPP_
#define TOPOLOGY_HPP_

#include <unordered_map>
#include <boost/fiber/numa/topology.hpp>
#include <iostream>

#include "common/gpu/gpu-common.hpp"

#include "nvml.h"

class gpu_id{
protected:
    uint32_t id;
};

class cpunumanode_id{
protected:
    uint32_t id;
};



/**
 * A describing the system topology of a single server.
 */
class topology{
private:
    class topologyonly_construction{
    private:
        topologyonly_construction(){}
        friend class topology;
    };

public:
    class cpunumanode{
    public:
        const uint32_t                      id              ;
        const uint32_t                      index_in_topo   ;


        std::vector<uint32_t>               distance        ;

        std::vector<uint32_t>               local_cores     ;
        cpu_set_t                           local_cpu_set   ;

    public:
        cpunumanode(const boost::fibers::numa::node &b, uint32_t index_in_topo,
                //do not remove argument!!! 
                topologyonly_construction);
    };

    class core{
    public:
        const uint32_t                      id              ;
        const uint32_t                      local_cpu       ;
        const uint32_t                      index_in_topo   ;

    public:
        core(uint32_t id, uint32_t local_cpu, uint32_t index_in_topo,
                //do not remove argument!!! 
                topologyonly_construction):
            id(id), local_cpu(local_cpu), index_in_topo(index_in_topo){}
    };

    class gpunode{
    public:
        const uint32_t                      id              ;
        const nvmlDevice_t                  handle          ;
        const uint32_t                      index_in_topo   ;

        std::vector<nvmlGpuTopologyLevel_t> connectivity    ;

        std::vector<uint32_t>               local_cores     ;
        cpu_set_t                           local_cpu_set   ;
        uint32_t                            local_cpu       ;

        // Use only if *absolutely* necessary!
        cudaDeviceProp                      properties      ;

    private:
        static nvmlDevice_t getGPUHandle(unsigned int id);

    public:
        gpunode(uint32_t id, uint32_t index_in_topo, 
                //do not remove argument!!! 
                topologyonly_construction);
    };

private:
    std::vector<cpunumanode >               cpu_info        ;
    std::vector<gpunode     >               gpu_info        ;

    std::vector<core        >               core_info       ;


    std::vector<uint32_t    >  cpunuma_index   ;

    uint32_t gpu_cnt ;
    uint32_t core_cnt;
protected:
    topology();

    static topology instance;
public:
    static inline const topology &getInstance(){
        return instance;
    }

    inline uint32_t getGpuCount() const{
        return gpu_cnt;
    }

    inline uint32_t getCoreCount() const{
        return core_cnt;
    }

    inline uint32_t getCpuNumaNodeCount() const{
        return cpu_info.size();
    }

    inline const std::vector<gpunode    >  &getGpus          () const{
        return gpu_info;
    }

    inline const std::vector<core       >  &getCores         () const{
        return core_info;
    }

    inline const std::vector<cpunumanode>  &getCpuNumaNodes  () const{
        return cpu_info;
    }

    inline const gpunode                   &getActiveGpu     () const{
        int device;
        gpu_run(cudaGetDevice(&device));
        return gpu_info[device];
    }

    template<typename T>
    const cpunumanode * getCpuNumaNodeAddressed(const T * m) const{
        int numa_id = -1;
        get_mempolicy(&numa_id, NULL, 0, (void *) m, MPOL_F_NODE | MPOL_F_ADDR);
        return &(getCpuNumaNodeById(numa_id));
    }

// int get_device(const void *p){
// #ifndef NCUDA
//     cudaPointerAttributes attrs;
//     cudaError_t error = cudaPointerGetAttributes(&attrs, p);
//     if (error == cudaErrorInvalidValue) return -1;
//     gpu_run(error);
//     return (attrs.memoryType == cudaMemoryTypeHost) ? -1 : attrs.device;
// #else
//     return -1;
// #endif
// }


    template<typename T>
    const gpunode * getGpuAddressed(const T * p) const{
#ifndef NCUDA
        cudaPointerAttributes attrs;
        cudaError_t error = cudaPointerGetAttributes(&attrs, p);
        if (error == cudaErrorInvalidValue) return NULL;
        gpu_run(error);
        if (attrs.memoryType == cudaMemoryTypeHost) return NULL;
        return &(getGpuByIndex(attrs.device));
#else
        return NULL;
#endif
    }

private:
    inline const gpunode &getGpuByIndex(uint32_t index) const{
        return gpu_info[index];
    }

    inline const cpunumanode &getCpuNumaNodeById(uint32_t id) const{
        return cpu_info[cpunuma_index[id]];
    }

    friend class exec_location;
    friend int numa_node_of_gpu(int device);
    friend std::ostream &operator<<(std::ostream& stream, const topology &topo);
};

std::ostream &operator<<(std::ostream &stream, const topology &topo);

// class local_topology{
    
// };

#endif /* TOPOLOGY_HPP_ */