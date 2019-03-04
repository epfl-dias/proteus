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

#ifndef AFFINITY_MANAGER_HPP_
#define AFFINITY_MANAGER_HPP_

#include "topology/topology.hpp"

template <typename T>
class buffer_manager;

/**
 * Really! do not touch this class!
 */
class affinity_cpu_set {
 private:
  static void set(const topology::cpunumanode &cpu, cpu_set_t cores);
  static cpu_set_t get();

  friend class exec_location;
  friend class affinity;
};

class affinity {
 private:
  static void set(const topology::cpunumanode &cpu);
  static void set(const topology::core &core);

  static const topology::cpunumanode &get();

  friend class exec_location;
  friend class MemoryManager;
  friend class NUMAMemAllocator;
  friend class NUMAPinnedMemAllocator;
  friend class buffer_manager<int32_t>;
};

class exec_location {
 private:
  int gpu_device;
  const topology::cpunumanode &cpu;
  const cpu_set_t cores;

 public:
  exec_location() : cpu(affinity::get()), cores(affinity_cpu_set::get()) {
    gpu_run(cudaGetDevice(&gpu_device));

    // CPU_ZERO(&cpus);
    // pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpus);
  }

  exec_location(int gpu);

  [[deprecated]] exec_location(int gpu, cpu_set_t cpus)
      : gpu_device(gpu),
        cpu(topology::getInstance().findCpuNumaNodes(cpus)),
        cores(cpus) {}
  // cpu(topology::getInstance().findCpuNumaNodes(cpus)){}

  [[deprecated]] exec_location(cpu_set_t cpus) : exec_location(-1, cpus) {}

  exec_location(const topology::core &core)
      : gpu_device(-1),
        cpu(topology::getInstance().getCpuNumaNodeById(core.local_cpu)),
        cores(core) {}

  exec_location(const topology::cpunumanode &cpu)
      : gpu_device(-1), cpu(cpu), cores(cpu.local_cpu_set) {}

  exec_location(const topology::gpunode &gpu);

 public:
  void activate() const {
    // std::cout << "d" << gpu_device << " " << cpu.local_cpu_set << std::endl;
    if (gpu_device >= 0) gpu_run(cudaSetDevice(gpu_device));
    // pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpus);
    affinity_cpu_set::set(cpu, cores);
    // std::this_thread::yield();
    // cpu_set_t d;
    // err = pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &d);
    // assert(err == 0);

    // std::cout << d << std::endl;
    // std::cout << sched_getcpu() << std::endl;
    // assert(CPU_ISSET(sched_getcpu(), &d));
  }
};

class set_device_on_scope {
 private:
  const topology::gpunode &device;

 public:
  inline set_device_on_scope(int set)
      : device(topology::getInstance().getActiveGpu()) {
    if (set >= 0) gpu_run(cudaSetDevice(set));
  }

  inline set_device_on_scope(const topology::gpunode &gpu)
      : device(topology::getInstance().getActiveGpu()) {
    gpu_run(cudaSetDevice(gpu.id));
  }

  inline ~set_device_on_scope() { gpu_run(cudaSetDevice(device.id)); }
};

class set_exec_location_on_scope {
 private:
  exec_location old;

 public:
  inline set_exec_location_on_scope(int gpu) { exec_location{gpu}.activate(); }

  inline set_exec_location_on_scope(int gpu, cpu_set_t cpus) {
    exec_location{gpu, cpus}.activate();
  }

  inline set_exec_location_on_scope(cpu_set_t cpus) {
    exec_location{cpus}.activate();
  }

  inline set_exec_location_on_scope(const exec_location &loc) {
    loc.activate();
  }

  inline set_exec_location_on_scope(const topology::cpunumanode &cpu) {
    exec_location{cpu}.activate();
  }

  inline set_exec_location_on_scope(const topology::core &core) {
    exec_location{core}.activate();
  }

  inline set_exec_location_on_scope(const topology::gpunode &gpu) {
    exec_location{gpu}.activate();
  }

  inline ~set_exec_location_on_scope() { old.activate(); }
};

int numa_node_of_gpu(int device);

#endif /* AFFINITY_MANAGER_HPP_ */
