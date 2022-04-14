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
#include <platform/common/gpu/gpu-common.hpp>
#include <platform/network/infiniband/devices/ib.hpp>
#include <unordered_map>
#include <vector>

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
    topologyonly_construction() = default;
    friend class topology;
  };

 public:
  class cu {
   public:
    inline explicit cu(
        // do not remove argument!!!
        topologyonly_construction = {}) {}

    // Do not allow copies
    cu(const cu &) = delete;
    cu &operator=(const cu &) = delete;

    // Allow construction through moving, but do not allow moving to overwrite
    cu(cu &&) noexcept = default;
    cu &operator=(cu &&) noexcept = delete;

    [[nodiscard]] virtual set_exec_location_on_scope set_on_scope() const = 0;
    virtual ~cu() = default;
  };

  class numanode : public cu {
   public:
    inline explicit numanode(
        // do not remove argument!!!
        topologyonly_construction = {}) {}

    // Do not allow copies
    numanode(const numanode &) = delete;
    numanode &operator=(const numanode &) = delete;

    // Allow construction through moving, but do not allow moving to overwrite
    numanode(numanode &&) noexcept = default;
    numanode &operator=(numanode &&) noexcept = delete;

    inline bool operator==(const numanode &other) const noexcept {
      return this == &other;
    }
  };

  class cpunumanode;

  class nvmeStorage : public numanode {
   public:
    const uint32_t id;          /// 32 bit identifier unique across nvme devices
    const std::string devPath;  /// linux device path, including nvme namespace
                                /// e.g /dev/nvme0n1
    const uint32_t
        index_in_topo;  /// index of nvme storage device in this topology
    const uint32_t
        local_cpu_id;  /// id of cpunumanode which this nvme is attached to
    const std::string model_name;  /// model name, human readable string
    const std::string link_speed;  /// link speed, string e.g 8 GT/s
    const uint32_t link_width;     /// link width, in number of PCIe lanes

   public:
    /**
     *
     * @param devPath path in /dev/{device name}, uniquely identifies a device,
     * but as a string is an inconvenient identifier. For nvme drives this is a
     * path to a namespace, e.g /dev/nvme0n1
     * @param index_in_topo index in topology vector, useful for identifying
     * within proteus quickly, but not stable across proteus instances
     */
    nvmeStorage(const std::string &devPath, uint32_t index_in_topo,
                // do not remove argument!!!
                topologyonly_construction = {});

    //     Do not allow copies
    nvmeStorage(const nvmeStorage &) = delete;
    nvmeStorage &operator=(const nvmeStorage &) = delete;

    // Allow construction through moving, but do not allow moving to overwrite
    nvmeStorage(nvmeStorage &&) = default;
    nvmeStorage &operator=(nvmeStorage &&) = delete;

    /**
     *
     * @return The cpunumanode to which this nvmeStorage belongs
     */
    const cpunumanode &getLocalCPUNumaNode() const;

    /**
     * Calling this function is undefined. See
     * https://gitlab.epfl.ch/DIAS/PROJECTS/caldera/proteus/-/issues/79 Calling
     * this function will likely (certainly) lead to a crash. It currently
     * exists because it is necessary to compile.
     * @return noreturn
     */
    [[nodiscard]] set_exec_location_on_scope set_on_scope()
        const override final;

    friend std::ostream &operator<<(std::ostream &out, const nvmeStorage &nvme);
  };

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

    /**
     * Moves execution of this scope to this cpunumanode
     * @return a set_exec_location_on_scope, code within the containing scope
     * will execute on this cpunumanode
     */
    [[nodiscard]] set_exec_location_on_scope set_on_scope()
        const override final;
  };

  class core : public cu {
   public:
    const uint32_t id;
    const uint32_t local_cpu_id;
    const uint32_t local_cpu_index;
    const uint32_t index_in_topo;
    const std::vector<uint32_t> ht_pairs_id;

   public:
    core(uint32_t id, uint32_t local_cpu, uint32_t index_in_topo,
         uint32_t local_cpu_index, const std::vector<uint32_t> ht_pairs,
         // do not remove argument!!!
         topologyonly_construction = {})
        : id(id),
          local_cpu_id(local_cpu),
          index_in_topo(index_in_topo),
          local_cpu_index(local_cpu_index),
          ht_pairs_id(ht_pairs) {}

    // const cpunumanode &getNumaNode() const;
    const cpunumanode &getLocalCPUNumaNode() const;

    /**
     * Moves execution of this scope to this cpu core
     * @return a set_exec_location_on_scope, code within the containing scope
     * will execute on this cpu core
     */
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
    uint32_t local_cpu_id;

    // Use only if *absolutely* necessary!
    cudaDeviceProp properties;

   private:
    static nvmlDevice_t getGPUHandle(unsigned int id);

   public:
    gpunode(uint32_t id, uint32_t index_in_topo,
            const std::vector<topology::core> &all_cores,
            // do not remove argument!!!
            topologyonly_construction = {});
    /**
     *
     * @return Total gloabl memory available on the device in bytes
     */
    size_t getMemorySize() const;

    /**
     *
     * @return the cpunumanode which is local to this gpunode
     */
    const cpunumanode &getLocalCPUNumaNode() const;

    /**
     * Moves execution of this scope to this gpunode
     * @return a set_exec_location_on_scope, code within the containing scope
     * will execute on this gpunode
     */
    [[nodiscard]] set_exec_location_on_scope set_on_scope()
        const override final;
  };

 private:
  std::vector<cpunumanode> cpu_info;
  std::vector<gpunode> gpu_info;
  std::vector<ib> ib_info;
  std::vector<nvmeStorage> nvmeStorage_info;

  std::vector<core> core_info;

  // mapping from numanode id to numanode index in topo
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

  /**
   * initialize nvme storage info, assumes that cpu topology info has been
   * initialized before calling
   */
  void init_nvmeStorage();

  /**
   * Maps a pcie address to a cpunumanode id
   * @param address string in the form '0000:04:00.0' as you may get from a cli
   * utility
   * @return a uint32_t corresponding to a cpunumanode id
   */
  uint32_t pcieAddressToNumaNodeId(std::string address);

  static topology instance;

 public:
  static void init();

  static inline const topology &getInstance() {
    assert(instance.getCoreCount() > 0 && "Is topology initialized?");
    return instance;
  }

  [[nodiscard]] inline uint32_t getGpuCount() const { return gpu_cnt; }

  [[nodiscard]] inline uint32_t getCoreCount() const { return core_cnt; }

  [[nodiscard]] inline uint32_t getCpuNumaNodeCount() const {
    return cpu_info.size();
  }

  [[nodiscard]] inline size_t getIBCount() const { return ib_info.size(); }

  [[nodiscard]] inline size_t getNvmeCount() const {
    return nvmeStorage_info.size();
  }

  [[nodiscard]] inline const std::vector<nvmeStorage> &getNvmes() const {
    return nvmeStorage_info;
  }

  [[nodiscard]] inline const std::vector<gpunode> &getGpus() const {
    return gpu_info;
  }

  [[nodiscard]] inline const std::vector<core> &getCores() const {
    return core_info;
  }

  [[nodiscard]] inline const std::vector<ib> &getIBs() const { return ib_info; }

  [[nodiscard]] inline const std::vector<cpunumanode> &getCpuNumaNodes() const {
    return cpu_info;
  }

  [[nodiscard]] inline const gpunode &getActiveGpu() const {
    int device = -1;
    gpu_run(cudaGetDevice(&device));
    return gpu_info[device];
  }

  /**
   *
   * @param m pointer to CPU memory
   * @return nullptr on failure, else pointer to the cpunumanode which contains
   * the memory this pointer points at
   */
  const cpunumanode *getCpuNumaNodeAddressed(const void *m) const;

  /**
   *
   * @param p pointer to GPU memory
   * @return nullptr on failure, else pointer to the gpunode which contains
   * the memory this pointer points at
   */
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

  const numanode &getNumaAddressed(const void *ptr) const {
    auto gpunode = getGpuAddressed(ptr);
    if (gpunode) return *gpunode;
    return *getCpuNumaNodeAddressed(ptr);
  }

  /**
   *
   * @return the cpunumanode which is local to this ib
   */
  [[nodiscard]] const topology::cpunumanode &findLocalCPUNumaNode(
      const ib &ib) const;

  /**
   *
   * @return the cpunumanode which is local to this nvmeStorage
   */
  [[nodiscard]] const topology::cpunumanode &findLocalCPUNumaNode(
      const nvmeStorage &nvme) const;

  inline const core &getCoreById(uint32_t id) const {
    return core_info[cpucore_index[id]];
  }

 private:
  [[deprecated]] inline const cpunumanode &findCpuNumaNodes(
      cpu_set_t cpus) const {
    for (const auto &t : getCpuNumaNodes()) {
      if (CPU_EQUAL(&t.local_cpu_set, &cpus)) return t;
    }
    throw std::runtime_error("unsupported affinity");
  }

  inline const gpunode &getGpuByIndex(uint32_t index) const {
    return gpu_info[index];
  }

 public:
  inline const cpunumanode &getCpuNumaNodeById(uint32_t id) const {
    return cpu_info[cpunuma_index[id]];
  }

  friend class exec_location;
  friend class affinity;
  friend class InfiniBandManager;
  friend int numa_node_of_gpu(int device);
  friend int node_of_gpu(int device);
  friend int get_rand_core_local_to_ptr(const void *p);
  friend std::ostream &operator<<(std::ostream &stream, const topology &topo);
  size_t getIBCount();
};

std::ostream &operator<<(std::ostream &stream, const topology &topo);
std::ostream &operator<<(std::ostream &out, const topology::nvmeStorage &nvme);
#endif /* TOPOLOGY_HPP_ */
