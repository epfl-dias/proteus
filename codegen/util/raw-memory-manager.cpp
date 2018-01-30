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

#include "util/raw-memory-manager.hpp"
#include "multigpu/buffer_manager.cuh"
#include "multigpu/numa_utils.cuh"
#include "nvToolsExt.h"

constexpr size_t freed_cache_cap      = 16;

void RawMemoryManager::init(){
    buffer_manager<int32_t>::init(128, 1024);
    int gpus = get_num_of_gpus();
    int cpus = numa_num_task_nodes();
    gpu_managers = new SingleGpuMemoryManager *[gpus];
    for (int device = 0 ; device < gpus ; ++device){
        set_device_on_scope d(device);
        gpu_managers[device] = new SingleGpuMemoryManager();
        //warm-up
        //NOTE: how many buffers should we warm up ? how do we calibrate that without cheating ?
        void * ptrs[8];
        for (size_t i = 0 ; i < 8 ; ++i){
            ptrs[i] = gpu_managers[device]->malloc((unit_capacity_gpu/4) * 3);
        }
        for (size_t i = 0 ; i < 8 ; ++i){
            gpu_managers[device]->free(ptrs[i]);
        }
    }
    cpu_managers = new SingleCpuMemoryManager *[cpus];
    for (int device = 0 ; device < cpus ; ++device){
        set_exec_location_on_scope d(cpu_numa_affinity[device]);
        cpu_managers[device] = new SingleCpuMemoryManager();
        //warm-up
        //NOTE: how many buffers should we warm up ? how do we calibrate that without cheating ?
        void * ptrs[8];
        for (size_t i = 0 ; i < 8 ; ++i){
            ptrs[i] = cpu_managers[device]->malloc((unit_capacity_cpu/4) * 3);
        }
        for (size_t i = 0 ; i < 8 ; ++i){
            cpu_managers[device]->free(ptrs[i]);
        }
    }
}

void RawMemoryManager::destroy(){
    buffer_manager<int32_t>::destroy();
    int gpus = get_num_of_gpus();
    for (int device = 0 ; device < gpus ; ++device){
        set_device_on_scope d(device);
        delete gpu_managers[device];
    }
    delete[] gpu_managers;
    int cpus = numa_num_task_nodes();
    for (int device = 0 ; device < cpus ; ++device){
        set_exec_location_on_scope d(cpu_numa_affinity[device]);
        delete cpu_managers[device];
    }
    delete[] cpu_managers;
}

constexpr inline size_t fixSize(size_t bytes){
    return (bytes + 0xF) & ~0xF;
}

void * RawMemoryManager::mallocGpu(size_t bytes){
    nvtxRangePushA("mallocGpu");
    bytes = fixSize(bytes);
    int dev = get_device();
    void * ptr = gpu_managers[dev]->malloc(bytes);
    nvtxRangePop();
    return ptr;
}

void   RawMemoryManager::freeGpu  (void * ptr){
    nvtxRangePushA("freeGpu");
    int dev = get_device(ptr);
    set_device_on_scope d(dev);
    gpu_managers[dev]->free(ptr);
    nvtxRangePop();
}

void * RawMemoryManager::mallocPinned(size_t bytes){
    nvtxRangePushA("mallocPinned");
    bytes = fixSize(bytes);
    int cpu  = sched_getcpu();
    int node = numa_node_of_cpu(cpu);
    void * ptr = cpu_managers[node]->malloc(bytes);
    nvtxRangePop();
    std::cout << "Alloc: " << node << " " << ptr << std::endl;
    return ptr;
}

void   RawMemoryManager::freePinned  (void * ptr){
    nvtxRangePushA("freePinned");
    int node;
    move_pages(0, 1, &ptr, NULL, &node, MPOL_MF_MOVE);
    std::cout << "Free: " << node << " " << ptr << std::endl;
    cpu_managers[node]->free(ptr);
    nvtxRangePop();
}

void * GpuMemAllocator::malloc(size_t bytes){
    void * ptr;
    gpu_run(cudaMalloc(&ptr, bytes));
    return ptr;
}

void GpuMemAllocator::free(void * ptr){
    gpu_run(cudaFree(ptr));
}

void * NUMAPinnedMemAllocator::malloc(size_t bytes){
    void *ptr = numa_alloc_onnode(bytes, numa_node_of_cpu(sched_getcpu()));
    assert(ptr && "Memory allocation failed!");
    gpu_run(cudaHostRegister(ptr, bytes, 0));
    sizes.emplace(ptr, bytes);
    return ptr;
}

void NUMAPinnedMemAllocator::free(void * ptr){
    gpu_run(cudaHostUnregister(ptr));
    auto it = sizes.find(ptr);
    assert(it != sizes.end() && "Memory did not originate from this allocator (or is already released)!");
    numa_free(ptr, it->second);
    sizes.erase(it);
}


template<typename allocator, size_t unit_cap>
SingleDeviceMemoryManager<allocator, unit_cap>::SingleDeviceMemoryManager(){}

template<typename allocator, size_t unit_cap>
SingleDeviceMemoryManager<allocator, unit_cap>::~SingleDeviceMemoryManager(){
    while (!free_cache.empty()){
        allocator::free(free_cache.top());
        free_cache.pop();
    }
    assert(allocations.empty());
    assert(mappings   .empty());
    assert(units      .empty());
    assert(big_units  .empty());
    assert(free_cache .empty());
}

template<typename allocator, size_t unit_cap>
typename SingleDeviceMemoryManager<allocator, unit_cap>::alloc_unit_info & SingleDeviceMemoryManager<allocator, unit_cap>::create_allocation(){
    void * ptr;
    if (free_cache.empty()){
        ptr = allocator::malloc(unit_cap);
    } else {
        ptr = free_cache.top();
        free_cache.pop();
    }
    allocations.emplace(ptr);
    return units.emplace(ptr, ptr).first->second;
}

template<typename allocator, size_t unit_cap>
void * SingleDeviceMemoryManager<allocator, unit_cap>::malloc(size_t bytes){
    bytes = fixSize(bytes);

    if (bytes >= unit_cap){
        void * ptr = allocator::malloc(bytes);

        {
            std::lock_guard<std::mutex> lock(m_big_units);
            big_units.emplace(ptr);
        }

        return ptr;
    }

    {
        std::lock_guard<std::mutex> lock(m);

        alloc_unit_info * info;
        if (allocations.empty()){
            info = &(create_allocation());
        } else {
            void * latest = allocations.top();

            auto              match = units.find(latest);
            assert(match != units.end() && "Unit not found!");
            info  = &(match->second);

            if (info->fill + bytes > unit_cap){
                info = &(create_allocation());
            }
        }

        void * ptr       = ((void *) (((char *) info->base) + info->fill));
        info->fill      += bytes;
        info->sub_units += 1    ;
        mappings.emplace(ptr, info->base);

        return ptr;
    }
}

template<typename allocator, size_t unit_cap>
void SingleDeviceMemoryManager<allocator, unit_cap>::free(void * ptr){
    {
        std::lock_guard<std::mutex> lock(m_big_units);
        auto f = big_units.find(ptr);
        if (f != big_units.end()){
            big_units.erase(f);
            allocator::free(ptr);
            return;
        }
    }

    {
        std::lock_guard<std::mutex> lock(m);
        auto f = mappings.find(ptr);
        if (f == mappings.end()){
            for (auto &t: mappings) std::cout << t.first << " " << t.second << std::endl;
        }
        assert(f != mappings.end() && "Mapping does not exist!");

        void            * base = f->second;
        mappings.erase(f);

        auto              fu   = units.find(base);
        assert(fu != units.end() && "Unit not found!");
        alloc_unit_info & info = fu->second;
        assert(info.sub_units > 0);
        info.sub_units         = info.sub_units - 1;
        if (info.sub_units == 0){
            if (!allocations.empty() && allocations.top() == base){
                allocations.pop();
                while (!allocations.empty()) {
                    void * tmp_base  = allocations.top();
                    bool still_valid = (units.find(tmp_base) != units.end());
                    if (!still_valid) allocations.pop();
                    else break;
                }
            }
            if (free_cache.size() < freed_cache_cap){
                free_cache.push(base);
            } else {
                allocator::free(ptr);
            }
            units.erase(fu);
        }
    }
}

SingleGpuMemoryManager ** RawMemoryManager::gpu_managers;
SingleCpuMemoryManager ** RawMemoryManager::cpu_managers;

std::unordered_map<void *, size_t> NUMAPinnedMemAllocator::sizes;