#include "topology/topology.hpp"
#include <iomanip>
#include <cmath>
#include <stdexcept>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "cuda_profiler_api.h"
#include "nvToolsExt.h"

#include "topology/affinity_manager.hpp"

topology::topology(){
    gpu_run(nvmlInit());

    // Creating gpunodes requires that we know the number of cores,
    // so start by reading the CPU configuration
    const auto &bcpu_info = boost::fibers::numa::topology();

    uint32_t max_numa_id = 0;
    for (auto &bcpu: bcpu_info) {
        cpu_info.emplace_back(bcpu, cpu_info.size(), 
            topologyonly_construction{});

        max_numa_id = std::max(max_numa_id, cpu_info.back().id);
    }
    
    cpunuma_index.resize(max_numa_id);
    for (auto &ci: cpunuma_index) ci = 0;

    for (size_t i = 0 ; i < cpu_info.size() ; ++i) {
        cpunuma_index[cpu_info[i].id] = i;
    }

    core_cnt = sysconf(_SC_NPROCESSORS_ONLN);
    assert(core_cnt > 0);
    
    for (const auto &cpu: cpu_info) {
        for (const auto &core: cpu.local_cores){
            core_info.emplace_back(core, cpu.id, core_info.size(), 
                topologyonly_construction{});
        }
    }

    assert(core_info.size() == core_cnt);

    unsigned int gpus = 0;
    gpu_run(cudaGetDeviceCount((int *) &gpus));
    gpu_cnt  = gpus;

    // Now create the GPU nodes
    for (uint32_t i = 0 ; i < gpu_cnt ; ++i) {
        gpu_info.emplace_back(i, i, topologyonly_construction{});
    }

    // warp-up GPUs
    for (const auto &gpu: gpu_info) {
        gpu_run(cudaSetDevice(gpu.id));
        gpu_run(cudaFree(0));
    }

    // P2P check & enable
    for (auto &gpu: gpu_info) {
        gpu.connectivity.resize(gpu_cnt);

        set_device_on_scope d(gpu.id);
        for (const auto &gpu2: gpu_info) {
            
            if (gpu2.id != gpu.id) {
                int t;
                gpu_run(cudaDeviceCanAccessPeer(&t, gpu.id, gpu2.id));
                if (t){
                    gpu_run(cudaDeviceEnablePeerAccess(gpu2.id, 0));
                } else {
                    std::cout << "Warning: P2P disabled for : GPU-" << gpu.id << " -> GPU-" << gpu2.id << std::endl;
                }

                gpu_run(nvmlDeviceGetTopologyCommonAncestor(gpu.handle, gpu2.handle, &(gpu.connectivity[gpu2.id])));
            }
        }
    }
}


std::ostream &operator<<(std::ostream &out, const topology &topo){
    out << "numa nodes: " << topo.getCpuNumaNodeCount   () << "\n";
    out << "core count: " << topo.getCoreCount          () << "\n";
    out << "gpu  count: " << topo.getGpuCount           () << "\n";

    out << '\n';

    char core_mask[topo.core_cnt + 1];
    core_mask[topo.core_cnt] = '\0';


    uint32_t digits = (uint32_t) std::ceil(std::log10(topo.core_cnt));

    for (uint32_t k = digits ; k > 0 ; --k){
        uint32_t base = std::pow(10, k - 1);

        if (k == ((digits+1)/2)) out << "core: " ;
        else                     out << "      " ;

        if      (1 == digits) out << ' ';
        else if (k == digits) out << '/';
        else if (k == 1     ) out << '\\';
        else                  out << '|';
        out << std::setw(base + 4 + 4 + 3 + 18) << ((k == 1) ? '0' : ' ');
        
        for (uint32_t i = base ; i < topo.core_cnt ; ++i) {
            out << (i / base) % 10;
        }
        out << '\n';
    }

    for (const auto &node: topo.cpu_info) {
        out << "node: " << std::setw(6) << node.id << " | ";

        out << std::setw(4 + 4 + 3) << ' ' << " | ";

        out << "cores: ";

        // for ( auto cpu_id : node.logical_cpus) {
        //     out << std::setw(4) << cpu_id << " ";
        // }

        for (uint32_t i = 0 ; i < topo.core_cnt ; ++i) core_mask[i] = ' ';

        for (auto cpu_id : node.local_cores) core_mask[cpu_id] = 'x';

        out << core_mask << '\n';
    }

    out << '\n';

    for (const auto &gpu: topo.gpu_info) {
        unsigned int nvml_ind;
        gpu_run(nvmlDeviceGetIndex(gpu.handle, &nvml_ind));
        out << "gpu : "  << std::setw(2) << gpu.id;
        out << std::setw(4) << ("(" + std::to_string(nvml_ind) + ")") << " | ";
        out << "node : " << std::setw(4) << gpu.local_cpu   << " | ";
        out << "cores: ";


        for (uint32_t i = 0 ; i < topo.core_cnt ; ++i) core_mask[i] = ' ';

        for (auto cpu_id : gpu.local_cores) core_mask[cpu_id] = 'x';

        out << core_mask << '\n';
        // for ( auto cpu_id : gpu.local_cores  ) {
        //     if (cpu_id)
        //     out << std::setw(4) << cpu_id << " ";
        // }
        // out << '\n';
    }

    // size_t sockets = topo.cpu_info.size();

    out << '\n';

    for (const auto &node: topo.cpu_info) {
        out << "node: ";
        out << node.id << " | ";
        for (auto d: node.distance) out << std::setw(4) << d;
        out << '\n';
    }

    out << '\n';

    for (const auto &gpu: topo.gpu_info) {
        out << "gpu : ";
        out << gpu.id          << " | ";
        for (auto d: gpu.connectivity) out << std::setw(4) << d;
        out << '\n';
    }
    return out;
}







topology::gpunode::gpunode(uint32_t id, uint32_t index_in_topo,
                            topologyonly_construction):
        id(id), handle(getGPUHandle(id)), index_in_topo(index_in_topo){
#ifndef NCUDA
    gpu_run(cudaGetDeviceProperties(&properties, id));

    const auto &topo = topology::getInstance();
    uint32_t sets = ((topo.getCoreCount() + 63) / 64);
    uint64_t cpuSet[sets];
    for (uint32_t i = 0 ; i < sets ; ++i) cpuSet[i] = 0;
    
    CPU_ZERO(&local_cpu_set);
    
    gpu_run(nvmlDeviceGetCpuAffinity(handle, sets, cpuSet));
    for (uint32_t i = 0 ; i < sets ; ++i){
        for (uint32_t k = 0 ; k < 64 ; ++k){
            if ((cpuSet[i] >> k) & 1){
                CPU_SET(64 * i + k, &(local_cpu_set));
            }
        }
    }

    uint32_t invalid = ~((uint32_t) 0);
    local_cpu = invalid;

    for (const auto &c: topo.getCores()){
        if (CPU_ISSET(c.id, &(local_cpu_set))) {
            local_cores.push_back(c.id);
            
            uint32_t cpu = c.local_cpu;
            assert(local_cpu == invalid || local_cpu == cpu);
            local_cpu = cpu;
        }
    }

    assert(local_cpu != invalid);
#else
    assert(false);
#endif
}

topology::cpunumanode::cpunumanode( const       boost::fibers::numa::node &b,
                                    uint32_t    index_in_topo,
                                    topologyonly_construction):
        id(b.id),
        distance(b.distance),
        local_cores(b.logical_cpus.begin(), b.logical_cpus.end()),
        index_in_topo(index_in_topo){
    CPU_ZERO(&local_cpu_set);
    for (const auto &c: local_cores) CPU_SET(c, &local_cpu_set);
}


nvmlDevice_t topology::gpunode::getGPUHandle(unsigned int id){
    cudaDeviceProp prop;
    gpu_run(cudaGetDeviceProperties(&prop, id));

    // NVML ignores CUDA_VISIBLE_DEVICES env variable, so we have to go over
    // all the available devices and find the one corresponding to the 
    // one referenced by @p id for the runtime api.
    // And, "yes, it's expected"
    // source: https://devtalk.nvidia.com/default/topic/815835/different-index-definition-in-nvml-amp-cuda-runtime-/

    unsigned int nvml_count = 0;
    gpu_run(nvmlDeviceGetCount(&nvml_count));
    // assert(device_count == gpus && "NMVL disagrees with cuda about the number of GPUs");


    // source: https://devblogs.nvidia.com/increase-performance-gpu-boost-k80-autoboost/
    for (unsigned int nvml_ind = 0; nvml_ind < nvml_count; ++nvml_ind){
        nvmlDevice_t d;
        gpu_run(nvmlDeviceGetHandleByIndex(nvml_ind, &d));
        
        nvmlPciInfo_t pcie_info;
        gpu_run(nvmlDeviceGetPciInfo(d, &pcie_info));
        
        if (static_cast<unsigned int>(prop.pciBusID   ) == pcie_info.bus    &&
            static_cast<unsigned int>(prop.pciDeviceID) == pcie_info.device &&
            static_cast<unsigned int>(prop.pciDomainID) == pcie_info.domain ){
            return d;
        }
    }
    throw new std::runtime_error("failed to locate device in nvml!");
}

topology topology::instance;


extern "C" {
    int get_rand_core_local_to_ptr(const void *p){
        const auto *dev = topology::getInstance().getGpuAddressed(p);
        if (dev) return dev->local_cores[rand() % dev->local_cores.size()];

        const auto *cpu = topology::getInstance().getCpuNumaNodeAddressed(p);
        return cpu->local_cores[rand() % cpu->local_cores.size()];
    }
}
