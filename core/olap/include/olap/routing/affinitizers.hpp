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

#ifndef AFFINITIZERS_HPP_
#define AFFINITIZERS_HPP_

#include <platform/topology/device-types.hpp>
#include <platform/topology/gpu-index.hpp>
#include <platform/topology/topology.hpp>

class Affinitizer {
 public:
  virtual ~Affinitizer() = default;
  virtual size_t getAvailableCUIndex(size_t i) const = 0;
  virtual const topology::cu &getAvailableCU(size_t i) const = 0;
  /**
   * Number of locality (NUMA) regions
   */
  virtual size_t size() const = 0;
  virtual size_t getLocalCUIndex(void *) const = 0;
};

std::unique_ptr<Affinitizer> getDefaultAffinitizer(DeviceType);

class AffinityPolicy {
 protected:
  std::vector<std::vector<size_t>> indexes;
  const Affinitizer *aff;

 public:
  AffinityPolicy(size_t fanout, const Affinitizer *aff);
  size_t getIndexOfRandLocalCU(void *ptr) const;
};

class CpuNumaNodeAffinitizer : public Affinitizer {
 protected:
  const topology::cu &getAvailableCU(size_t cpu_req) const override {
    return topology::getInstance()
        .getCpuNumaNodes()[cpu_req %
                           topology::getInstance().getCpuNumaNodeCount()];
  }

 public:
  size_t getAvailableCUIndex(size_t i) const override {
    return dynamic_cast<const topology::cpunumanode &>(getAvailableCU(i))
        .index_in_topo;
  }

  size_t size() const override {
    return topology::getInstance().getCpuNumaNodeCount();
  }

  size_t getLocalCUIndex(void *p) const override {
    auto &topo = topology::getInstance();
    const auto *g = topo.getGpuAddressed(p);
    if (g) return g->getLocalCPUNumaNode().index_in_topo;
    auto *c = topo.getCpuNumaNodeAddressed(p);
    assert(c);
    return c->index_in_topo;
  }
};

class GPUAffinitizer : public Affinitizer {
 protected:
  const topology::gpunode &getAvailableCU(size_t gpu_req) const override {
    static const gpu_index index;
    size_t gpu_i = gpu_req % topology::getInstance().getGpuCount();
    return topology::getInstance().getGpus()[index.d[gpu_i]];
  }

 public:
  size_t getAvailableCUIndex(size_t i) const override {
    return getAvailableCU(i).index_in_topo;
  }

  size_t size() const override { return topology::getInstance().getGpuCount(); }

  size_t getLocalCUIndex(void *p) const override {
    auto &topo = topology::getInstance();
    const auto *g = topo.getGpuAddressed(p);
    if (g) return g->index_in_topo;
    auto *c = topo.getCpuNumaNodeAddressed(p);
    if (!c) {
      return topology::getInstance().getGpus()[rand() % size()].index_in_topo;
    }
    assert(c);
    const auto &gpus = c->local_gpus;
    return gpus[rand() % gpus.size()];
  }
};

class CpuCoreAffinitizer : public CpuNumaNodeAffinitizer {
 protected:
  virtual const topology::core &getAvailableCore(size_t cpu_req) const {
    size_t core_index = cpu_req % topology::getInstance().getCoreCount();
    // NOTE: Assuming all CPUs have the same number of cores!
    size_t cpunumacnt = topology::getInstance().getCpuNumaNodeCount();
    size_t numanode = core_index % cpunumacnt;
    const auto &cpunumanode =
        topology::getInstance().getCpuNumaNodes()[numanode];
    return cpunumanode.getCore(core_index / cpunumacnt);
  }

 protected:
  const topology::cu &getAvailableCU(size_t cpu_req) const final {
    return getAvailableCore(cpu_req);
  }

  size_t getAvailableCUIndex(size_t i) const override {
    return getAvailableCore(i).getLocalCPUNumaNode().index_in_topo;
  }
};

class SpecificCpuCoreAffinitizer : public CpuCoreAffinitizer {
 public:
  typedef std::remove_cv_t<decltype(topology::core::id)> coreid_t;

 private:
  std::vector<coreid_t> core_ids;

 public:
  SpecificCpuCoreAffinitizer(const std::vector<coreid_t> &core_ids)
      : core_ids(core_ids) {}

 protected:
  const topology::core &getAvailableCore(size_t cpu_req) const override {
    assert(cpu_req < core_ids.size());
    return topology::getInstance().getCoreById(core_ids[cpu_req]);
  }
};
#endif /* AFFINITIZERS_HPP_ */
