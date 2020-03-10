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

#ifndef TOPOLOGY_HPP_
#define TOPOLOGY_HPP_

#include <iostream>
#include <unordered_map>
#include <vector>

#include "common/gpu/gpu-common.hpp"
#include "nvml.h"

struct ibv_device;
class set_exec_location_on_scope;

/**
 * A describing the system topology of a single server.
 */
class topology {
 private:
  class topologyonly_construction {
   private:
    topologyonly_construction() {}
    friend class topology;
  };

 public:
  class cu {
   public:
    [[nodiscard]] virtual set_exec_location_on_scope set_on_scope() const = 0;
    virtual ~cu() = default;
  };

  class numanode : public cu {};

  class core;

  class cpunumanode : public numanode {
   public:
    const uint32_t id;
    const uint32_t index_in_topo;

    std::vector<uint32_t> distance;

    std::vector<uint32_t> local_cores;
    std::vector<uint32_t> local_gpus;
    cpu_set_t local_cpu_set;

   public:
    cpunumanode(uint32_t id, const std::vector<uint32_t> &local_cores,
                uint32_t index_in_topo,
                // do not remove argument!!!
                topologyonly_construction = {});

    // Do not allow copies
    cpunumanode(const cpunumanode &) = delete;
    cpunumanode &operator=(const cpunumanode &) = delete;

    // Allow construction through moving, but do not allow moving to overwrite
    cpunumanode(cpunumanode &&) = default;
    cpunumanode &operator=(cpunumanode &&) = delete;

    void *alloc(size_t bytes) const;
    static void free(void *mem, size_t bytes);
    size_t getMemorySize() const;

    const core &getCore(size_t i) const {
      assert(i < local_cores.size());
      return topology::getInstance().getCoreById(local_cores[i]);
    }

    [[nodiscard]] set_exec_location_on_scope set_on_scope()
        const override final;
  };

  class core : public cu {
   public:
    const uint32_t id;
    const uint32_t local_cpu;
    const uint32_t local_cpu_index;
    const uint32_t index_in_topo;

   public:
    core(uint32_t id, uint32_t local_cpu, uint32_t index_in_topo,
         uint32_t local_cpu_index,
         // do not remove argument!!!
         topologyonly_construction = {})
        : id(id),
          local_cpu(local_cpu),
          index_in_topo(index_in_topo),
          local_cpu_index(local_cpu_index) {}

    // const cpunumanode &getNumaNode() const;
    const cpunumanode &getLocalCPUNumaNode() const;

    [[nodiscard]] set_exec_location_on_scope set_on_scope()
        const override final;

   private:
    operator cpu_set_t() const {
      cpu_set_t tmp;
      CPU_ZERO(&tmp);
      CPU_SET(id, &tmp);
      return tmp;
    }

    friend class exec_location;
  };

  class gpunode : public numanode {
   public:
    const uint32_t id;
    const nvmlDevice_t handle;
    const uint32_t index_in_topo;

    std::vector<nvmlGpuTopologyLevel_t> connectivity;

    std::vector<uint32_t> local_cores;
    cpu_set_t local_cpu_set;
    uint32_t local_cpu;

    // Use only if *absolutely* necessary!
    cudaDeviceProp properties;

   private:
    static nvmlDevice_t getGPUHandle(unsigned int id);

   public:
    gpunode(uint32_t id, uint32_t index_in_topo,
            const std::vector<topology::core> &all_cores,
            // do not remove argument!!!
            topologyonly_construction = {});
    size_t getMemorySize() const;
    const cpunumanode &getLocalCPUNumaNode() const;

    [[nodiscard]] set_exec_location_on_scope set_on_scope()
        const override final;
  };

 private:
  std::vector<cpunumanode> cpu_info;
  std::vector<gpunode> gpu_info;

  std::vector<core> core_info;

  std::vector<uint32_t> cpunuma_index;
  std::vector<uint32_t> cpucore_index;

  uint32_t gpu_cnt;
  uint32_t core_cnt;

 protected:
  topology(topology &&) = delete;
  topology(const topology &) = delete;
  topology &operator=(topology &&) = delete;
  topology &operator=(const topology &) = delete;
  topology();
  void init_();

  static topology instance;

 public:
  static void init();

  static inline const topology &getInstance() {
    assert(instance.getCoreCount() > 0 && "Is topology initialized?");
    return instance;
  }

  inline uint32_t getGpuCount() const { return gpu_cnt; }

  inline uint32_t getCoreCount() const { return core_cnt; }

  inline uint32_t getCpuNumaNodeCount() const { return cpu_info.size(); }

  inline const std::vector<gpunode> &getGpus() const { return gpu_info; }

  inline const std::vector<core> &getCores() const { return core_info; }

  inline const std::vector<cpunumanode> &getCpuNumaNodes() const {
    return cpu_info;
  }

  inline const gpunode &getActiveGpu() const {
    int device = -1;
    gpu_run(cudaGetDevice(&device));
    return gpu_info[device];
  }

  const cpunumanode *getCpuNumaNodeAddressed(const void *m) const;

  [[deprecated]] uint32_t getCpuNumaNodeOfCore(uint32_t core_id) const {
    return core_info[core_id].local_cpu;
  }

  template <typename T>
  const gpunode *getGpuAddressed(const T *p) const {
#ifndef NCUDA
    if (getGpuCount() == 0) return nullptr;
    cudaPointerAttributes attrs;
    cudaError_t error = cudaPointerGetAttributes(&attrs, p);
    if (error == cudaErrorInvalidValue) return nullptr;
    gpu_run(error);
    if (attrs.type != cudaMemoryTypeDevice) return nullptr;
    return &(getGpuByIndex(attrs.device));
#else
    return nullptr;
#endif
  }

  const topology::cpunumanode &findLocalCPUNumaNode(ibv_device *ib_dev) const;

  inline const core &getCoreById(uint32_t id) const {
    return core_info[cpucore_index[id]];
  }

 private:
  [[deprecated]] inline const cpunumanode &findCpuNumaNodes(
      cpu_set_t cpus) const {
    for (const auto &t : getCpuNumaNodes()) {
      if (CPU_EQUAL(&t.local_cpu_set, &cpus)) return t;
    }
    throw new std::runtime_error("unsupported affinity");
  }

  inline const gpunode &getGpuByIndex(uint32_t index) const {
    return gpu_info[index];
  }

  inline const cpunumanode &getCpuNumaNodeById(uint32_t id) const {
    return cpu_info[cpunuma_index[id]];
  }

  friend class exec_location;
  friend class affinity;
  friend int numa_node_of_gpu(int device);
  friend int get_rand_core_local_to_ptr(const void *p);
  friend std::ostream &operator<<(std::ostream &stream, const topology &topo);
};

std::ostream &operator<<(std::ostream &stream, const topology &topo);

#endif /* TOPOLOGY_HPP_ */
