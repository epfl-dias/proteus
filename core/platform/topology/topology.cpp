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

#include "topology/topology.hpp"

#include <numa.h>
#include <numaif.h>

#include <cmath>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <vector>

#include "common/error-handling.hpp"
#include "cuda.h"
#include "cuda_profiler_api.h"
#include "cuda_runtime_api.h"
#include "nvToolsExt.h"
#include "nvml.h"
#include "topology/affinity_manager.hpp"

// template<typename T>
const topology::cpunumanode *topology::getCpuNumaNodeAddressed(
    const void *m) const {
  int numa_id = -1;
#ifndef NDEBUG
  long ret =
#endif
      get_mempolicy(&numa_id, nullptr, 0, const_cast<void *>(m),
                    MPOL_F_NODE | MPOL_F_ADDR);
  assert(ret == 0);
  assert(numa_id >= 0);
  return (cpu_info.data() + cpunuma_index[numa_id]);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
extern "C" void numa_error(char *where) { LOG(FATAL) << where; }
extern "C" void numa_warn(int num, char *fmt, ...) { LOG(WARNING) << fmt; }
#pragma clang diagnostic pop

void *topology::cpunumanode::alloc(size_t bytes) const {
  constexpr size_t hugepage = 2 * 1024 * 1024;
  bytes = ((bytes + hugepage - 1) / hugepage) * hugepage;
  void *mem = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
  assert(mem != MAP_FAILED);
  assert((((uintptr_t)mem) % hugepage) == 0);
  linux_run(madvise(mem, bytes, MADV_DONTFORK));
#ifndef NDEBUG
  {
    int status;
    // use move_pages as getCpuNumaNodeAddressed checks only the policy
    assert(move_pages(0, 1, &mem, nullptr, &status, 0) == 0);
    // check that page has not been prefaulted (status should be -ENOENT)!
    // otherwise, setting the numa policy will not be effective
    assert(status == -ENOENT);
  }
#endif

  // TODO: consider using numa_set_strict
  numa_tonode_memory(mem, bytes, id);

#ifndef NDEBUG
  {
    int status;
    // use move_pages as getCpuNumaNodeAddressed checks only the policy
    assert(move_pages(0, 1, &mem, nullptr, &status, 0) == 0);
    // That check is not critical but only a sanity check, consider removing
    assert(status == -ENOENT);
  }

  if (bytes >= sizeof(int)) {
    // fault first page to check it's allocated on the correct node
    ((int *)mem)[0] = 0;  // SIGBUS on this line means we run out of huge pafes
                          // in that CPU numa node
    // check using the policy
    assert(topology::getInstance().getCpuNumaNodeAddressed(mem) == this);

    {
      // now the first page should have been prefaulted, verify using move_pages
      int status;
      assert(move_pages(0, 1, &mem, nullptr, &status, 0) == 0);
      // the faulted page should be on the correct socket now
      assert(status == id);
    }
  }
#endif

  return mem;
  //  return numa_alloc_onnode(bytes, id);
}

void topology::cpunumanode::free(void *mem, size_t bytes) {
  // numa_free(mem, bytes);
  munmap(mem, bytes);
}

size_t topology::cpunumanode::getMemorySize() const {
  return numa_node_size64(id, nullptr);
}

void topology::init() {
  instance.init_();
  std::cout << topology::getInstance() << std::endl;
}

void topology::init_() {
  // Check if topology is already initialized and if yes, return early
  // This should only happen when a unit-test is reinitializing proteus but it
  // should not happen in normal execution, except if we "restart" proteus
  if (core_info.size() > 0) {
#ifndef NDEBUG
    auto core_cnt = sysconf(_SC_NPROCESSORS_ONLN);
    assert(core_info.size() == core_cnt);
#endif
    return;
  }
  assert(cpu_info.size() == 0 && "Is topology already initialized?");
  assert(core_info.size() == 0 && "Is topology already initialized?");
  unsigned int gpus = 0;
  auto nvml_res = nvmlInit();
  if (nvml_res == NVML_SUCCESS) {
    // We can not use gpu_run(...) before we set gpu_cnt, call gpuAssert
    // directly.
    gpuAssert(cudaGetDeviceCount((int *)&gpus), __FILE__, __LINE__);
  }
  gpu_cnt = gpus;

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

  cpucore_index.resize(core_cnt);
  for (const auto &cpu : cpu_info) {
    for (const auto &core : cpu.local_cores) {
      cpucore_index[core] = core_info.size();
      core_info.emplace_back(core, cpu.id, core_info.size());
    }
  }

  assert(core_info.size() == core_cnt);

  // Now create the GPU nodes
  for (uint32_t i = 0; i < gpu_cnt; ++i) {
    gpu_info.emplace_back(i, i, core_info);
    const auto &ind = cpunuma_index[gpu_info.back().local_cpu];
    cpu_info[ind].local_gpus.push_back(i);
  }

  // warp-up GPUs
  for (const auto &gpu : gpu_info) {
    gpu_run(cudaSetDevice(gpu.id));
    gpu_run(cudaFree(nullptr));
  }

  // P2P check & enable
  for (auto &gpu : gpu_info) {
    gpu.connectivity.resize(gpu_cnt);

    gpu_run(cudaSetDevice(gpu.id));
    // set_device_on_scope d(gpu);
    for (const auto &gpu2 : gpu_info) {
      if (gpu2.id != gpu.id) {
        int t = 0;
        gpu_run(cudaDeviceCanAccessPeer(&t, gpu.id, gpu2.id));
        if (t) {
          gpu_run(cudaDeviceEnablePeerAccess(gpu2.id, 0));
        } else {
          std::cout << "Warning: P2P disabled for : GPU-" << gpu.id;
          std::cout << " -> GPU-" << gpu2.id << std::endl;
        }

        gpu_run(nvmlDeviceGetTopologyCommonAncestor(
            gpu.handle, gpu2.handle, &(gpu.connectivity[gpu2.id])));
      }
    }
  }
}

topology::topology() {}

std::ostream &operator<<(std::ostream &out, const cpu_set_t &cpus) {
  long cores = sysconf(_SC_NPROCESSORS_ONLN);

  bool printed = false;

  for (int i = 0; i < cores; ++i)
    if (CPU_ISSET(i, &cpus)) {
      if (printed) out << ",";
      printed = true;
      out << i;
    }

  return out;
}

std::ostream &operator<<(std::ostream &out, const topology &topo) {
  out << "numa nodes: " << topo.getCpuNumaNodeCount() << "\n";
  out << "core count: " << topo.getCoreCount() << "\n";
  out << "gpu  count: " << topo.getGpuCount() << "\n";

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

  for (const auto &gpu : topo.getGpus()) {
    unsigned int nvml_ind = 0;
    gpu_run(nvmlDeviceGetIndex(gpu.handle, &nvml_ind));
    out << "gpu : " << std::setw(2) << gpu.id;
    out << std::setw(4) << ("(" + std::to_string(nvml_ind) + ")") << " | ";
    out << "node : " << std::setw(4) << gpu.local_cpu << " | ";
    out << "cores: ";

    for (uint32_t i = 0; i < topo.core_cnt; ++i) core_mask[i] = ' ';

    for (auto cpu_id : gpu.local_cores) core_mask[cpu_id] = 'x';

    out << core_mask << " | name: " << gpu.properties.name << '\n';
    // for ( auto cpu_id : gpu.local_cores  ) {
    //     if (cpu_id)
    //     out << std::setw(4) << cpu_id << " ";
    // }
    // out << '\n';
  }

  // size_t sockets = topo.cpu_info.size();

  out << '\n';

  for (const auto &node : topo.getCpuNumaNodes()) {
    out << "node: ";
    out << node.id << " | ";
    for (auto d : node.distance) out << std::setw(4) << d;
    out << '\n';
  }

  out << '\n';

  for (const auto &gpu : topo.getGpus()) {
    out << "gpu : " << gpu.id << " | ";
    for (auto d : gpu.connectivity) out << std::setw(4) << d;
    out << '\n';
  }
  return out;
}

topology::gpunode::gpunode(uint32_t id, uint32_t index_in_topo,
                           const std::vector<topology::core> &all_cores,
                           topologyonly_construction)
    : id(id), handle(getGPUHandle(id)), index_in_topo(index_in_topo) {
#ifndef NCUDA
  gpu_run(cudaGetDeviceProperties(&properties, id));

  uint32_t sets = ((all_cores.size() + 63) / 64);
  uint64_t cpuSet[sets];
  for (uint32_t i = 0; i < sets; ++i) cpuSet[i] = 0;

  CPU_ZERO(&local_cpu_set);

  gpu_run(nvmlDeviceGetCpuAffinity(handle, sets, cpuSet));
  for (uint32_t i = 0; i < sets; ++i) {
    for (uint32_t k = 0; k < 64; ++k) {
      if ((cpuSet[i] >> k) & 1) CPU_SET(64 * i + k, &(local_cpu_set));
    }
  }

  uint32_t invalid = ~((uint32_t)0);
  uint32_t tmp_cpu = invalid;

  for (const auto &c : all_cores) {
    if (CPU_ISSET(c.id, &(local_cpu_set))) {
      local_cores.push_back(c.id);

      uint32_t cpu = c.local_cpu;
      assert(tmp_cpu == invalid || tmp_cpu == cpu);
      tmp_cpu = cpu;
    }
  }

  assert(tmp_cpu != invalid);
  local_cpu = tmp_cpu;
#else
  assert(false);
#endif
}

size_t topology::gpunode::getMemorySize() const {
  return properties.totalGlobalMem;
}

topology::cpunumanode::cpunumanode(uint32_t id,
                                   const std::vector<uint32_t> &local_cores,
                                   uint32_t index_in_topo,
                                   topologyonly_construction)
    : id(id),
      // distance(b.distance),
      index_in_topo(index_in_topo),
      local_cores(local_cores) {
  CPU_ZERO(&local_cpu_set);
  for (const auto &c : local_cores) CPU_SET(c, &local_cpu_set);
}

nvmlDevice_t topology::gpunode::getGPUHandle(unsigned int id) {
  cudaDeviceProp prop;
  gpu_run(cudaGetDeviceProperties(&prop, id));

  // NVML ignores CUDA_VISIBLE_DEVICES env variable, so we have to go over
  // all the available devices and find the one corresponding to the
  // one referenced by @p id for the runtime api.
  // And, "yes, it's expected", source:
  // https://devtalk.nvidia.com/default/topic/815835/different-index-definition-in-nvml-amp-cuda-runtime-/

  unsigned int nvml_count = 0;
  gpu_run(nvmlDeviceGetCount(&nvml_count));
  // assert(device_count == gpus &&
  //        "NMVL disagrees with cuda about the number of GPUs");

  // source:
  // https://devblogs.nvidia.com/increase-performance-gpu-boost-k80-autoboost/
  for (unsigned int nvml_ind = 0; nvml_ind < nvml_count; ++nvml_ind) {
    nvmlDevice_t d;
    gpu_run(nvmlDeviceGetHandleByIndex(nvml_ind, &d));

    nvmlPciInfo_t pcie_info;
    gpu_run(nvmlDeviceGetPciInfo(d, &pcie_info));

    if (static_cast<unsigned int>(prop.pciBusID) == pcie_info.bus &&
        static_cast<unsigned int>(prop.pciDeviceID) == pcie_info.device &&
        static_cast<unsigned int>(prop.pciDomainID) == pcie_info.domain) {
      return d;
    }
  }
  throw new std::runtime_error("failed to locate device in nvml!");
}

topology topology::instance;

extern "C" int get_rand_core_local_to_ptr(const void *p) {
  // const auto *dev = topology::getInstance().getGpuAddressed(p);
  // if (dev) return dev->local_cores[rand() % dev->local_cores.size()];
  // const auto *cpu = topology::getInstance().getCpuNumaNodeAddressed(p);
  // return cpu->local_cores[rand() % cpu->local_cores.size()];

  // actually, for the current exchange implementation we should return
  // the integer i such that (i % #gpus) is a _gpu_ local to the current
  // numa node addressed. (and yes, this will cause problems on machines
  // without GPUs, but such machines need issue #16 to be resolved)
  // FIXME: related to issue #16 and the above comment
  // FIXME: *up*

  const auto &topo = topology::getInstance();
  const auto gpu_count = topo.getGpuCount();
  const auto *dev = topology::getInstance().getGpuAddressed(p);
  if (dev) return dev->id + ((rand() / gpu_count) * gpu_count);

  const auto *cpu = topology::getInstance().getCpuNumaNodeAddressed(p);

  const auto &local_gpus = cpu->local_gpus;
  size_t local_gpu_count = local_gpus.size();
  if (local_gpu_count == 0) return rand();

  const auto &sdev = local_gpus[rand() % local_gpu_count];
  return sdev + ((rand() / gpu_count) * gpu_count);
}

extern "C" int rand_local_cpu(const void *p, uint64_t fanout) {
  const auto *g = topology::getInstance().getGpuAddressed(p);
  if (g) assert(false && "TODO");
  const auto *c = topology::getInstance().getCpuNumaNodeAddressed(p);
  assert(c);
  size_t socket = c->index_in_topo;
  size_t nsockets = topology::getInstance().getCpuNumaNodeCount();
  size_t ulimit = (fanout - 1 - socket) / nsockets;
  size_t r = rand() % ulimit;
  return socket + r * nsockets;
}

const topology::cpunumanode &topology::gpunode::getLocalCPUNumaNode() const {
  return topology::getInstance().getCpuNumaNodeById(local_cpu);
}

const topology::cpunumanode &topology::core::getLocalCPUNumaNode() const {
  return topology::getInstance().getCpuNumaNodeById(local_cpu);
}

set_exec_location_on_scope topology::cpunumanode::set_on_scope() const {
  return {*this};
}

set_exec_location_on_scope topology::core::set_on_scope() const {
  return {*this};
}

set_exec_location_on_scope topology::gpunode::set_on_scope() const {
  return {*this};
}
