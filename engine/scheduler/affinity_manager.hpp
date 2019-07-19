/*
                  AEOLUS - In-Memory HTAP-Ready OLTP Engine

                              Copyright (c) 2019-2019
           Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

                              All Rights Reserved.

      Permission to use, copy, modify and distribute this software and its
    documentation is hereby granted, provided that both the copyright notice
  and this permission notice appear in all copies of the software, derivative
  works or modified versions, and any portions thereof, and that both notices
                      appear in supporting documentation.

  This code is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 A PARTICULAR PURPOSE. THE AUTHORS AND ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER RESULTING FROM THE
                             USE OF THIS SOFTWARE.
*/

#ifndef SCHEDULER_AFFINITY_MANAGER_HPP_
#define SCHEDULER_AFFINITY_MANAGER_HPP_

#include "scheduler/topology.hpp"

namespace scheduler {
// TODO: EVERYTHING
class AffinityManager {
 protected:
 public:
  // Singleton
  static inline AffinityManager &getInstance() {
    static AffinityManager instance;
    return instance;
  }
  AffinityManager(AffinityManager const &) = delete;  // Don't Implement
  void operator=(AffinityManager const &) = delete;   // Don't implement

  void set(const core *core);
  void set(const cpunumanode *cpu);

 private:
  AffinityManager() {}
};

/*
class affinity_cpu_set {
 private:
  static void set(const cpunumanode &cpu, cpu_set_t cores);
  static cpu_set_t get();

  friend class exec_location;
  friend class affinity;
};

class affinity {
 private:
  static void set(const cpunumanode &cpu);
  static void set(const core &core);

  static const cpunumanode &get();

  friend class exec_location;
  friend class MemoryManager;
  friend class NUMAPinnedMemAllocator;
};

class exec_location {
 private:
  const cpunumanode &cpu;
  const cpu_set_t cores;

 public:
  exec_location(const core &core)
      : cpu(Topology::getInstance().getCpuNumaNodeById(core.local_cpu)),
        cores(core) {}

  exec_location(const cpunumanode &cpu) : cpu(cpu), cores(cpu.local_cpu_set) {}

 public:
  void activate() const { affinity_cpu_set::set(cpu, cores); }
};

class set_exec_location_on_scope {
 private:
  exec_location old;

 public:
  inline set_exec_location_on_scope(cpu_set_t cpus) {
    exec_location{cpus}.activate();
  }

  inline set_exec_location_on_scope(const exec_location &loc) {
    loc.activate();
  }

  inline set_exec_location_on_scope(const cpunumanode &cpu) {
    exec_location{cpu}.activate();
  }

  inline set_exec_location_on_scope(const core &core) {
    exec_location{core}.activate();
  }

  inline ~set_exec_location_on_scope() { old.activate(); }
};*/

}  // namespace scheduler

#endif /* SCHEDULER_AFFINITY_MANAGER_HPP_ */
