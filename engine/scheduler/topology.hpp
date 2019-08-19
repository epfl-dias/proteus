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

#ifndef SCHEDULER_TOPOLOGY_HPP_
#define SCHEDULER_TOPOLOGY_HPP_

#include <sched.h>
#include <iostream>
#include <thread>
#include <vector>

namespace scheduler {

// class cpunumanode;
class Topology;
class core;
class cpunumanode;

std::ostream &operator<<(std::ostream &out, const Topology &topo);

class Topology {
 public:
  // Singleton
  static inline Topology &getInstance() {
    static Topology instance;
    return instance;
  }
  Topology(Topology const &) = delete;        // Don't Implement
  void operator=(Topology const &) = delete;  // Don't implement

  int get_num_worker_cores() { return core_info.size(); }
  std::vector<core> *get_worker_cores(int num = -1) { return &core_info; }

  inline uint32_t getCoreCount() const { return core_cnt; }

  inline uint32_t getCpuNumaNodeCount() const { return cpu_info.size(); }

  inline const std::vector<core> &getCores() const { return core_info; }

  inline const cpunumanode &getCpuNumaNodeById(uint32_t id) const {
    return cpu_info[cpunuma_index[id]];
  }
  inline const std::vector<cpunumanode> &getCpuNumaNodes() const {
    return cpu_info;
  }

 private:
  Topology();

  std::vector<cpunumanode> cpu_info;
  std::vector<core> core_info;
  std::vector<uint32_t> cpunuma_index;
  uint32_t core_cnt;
  friend std::ostream &operator<<(std::ostream &out, const Topology &topo);
};

class cpunumanode {
 public:
  const uint32_t id;
  const uint32_t index_in_topo;

  std::vector<uint32_t> distance;

  std::vector<uint32_t> local_cores;
  std::vector<uint32_t> local_gpus;
  cpu_set_t local_cpu_set;

 public:
  cpunumanode(uint32_t id, const std::vector<uint32_t> &local_cores,
              uint32_t index_in_topo);

  void *alloc(size_t bytes) const;
  static void free(void *mem, size_t bytes);
};

class core {
 public:
  const uint32_t id;
  const uint32_t local_cpu;
  const uint32_t index_in_topo;

 public:
  core(uint32_t id, uint32_t local_cpu, uint32_t index_in_topo)
      : id(id), local_cpu(local_cpu), index_in_topo(index_in_topo) {}

  // const cpunumanode &getNumaNode() const;
 private:
  operator cpu_set_t() const {
    cpu_set_t tmp;
    CPU_ZERO(&tmp);
    CPU_SET(id, &tmp);
    return tmp;
  }
};

}  // namespace scheduler

#endif /* SCHEDULER_TOPOLOGY_HPP_ */
