/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2023
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

#ifndef PROTEUS_WORKER_SCHEDULE_POLICY_HPP
#define PROTEUS_WORKER_SCHEDULE_POLICY_HPP

#include <platform/topology/topology.hpp>
#include <platform/util/erase-constructor-idioms.hpp>

#include "oltp/common/common.hpp"

namespace scheduler {

enum WorkerSchedulePolicy {
  PHYSICAL_ONLY,                // use physical threads only across NUMA
  PHYSICAL_FIRST,               // first use physical, then HT across NUMA
  SOCKET_FIRST_PHYSICAL_FIRST,  // first physical, then HT, before next NUMA
  CORE_FIRST,                   // physical + HT of same core before next core
  HALF_SOCKET_PHYSICAL,  // half of one socket, and then half of 2nd socket
  HALF_SOCKET,           // fill half socket with OLTP engine
  INTERLEAVE_EVEN,       // Schedule on even cores 0, 2, 4...
  INTERLEAVE_ODD         // Schedule on odd cores 1, 3, 5...
};

std::ostream& operator<<(std::ostream& out, WorkerSchedulePolicy policy);

using core_id_t = uint32_t;

class ScheduleWorkers : proteus::utils::remove_copy_move {
 public:
  static ScheduleWorkers& getInstance() {
    static ScheduleWorkers instance;
    return instance;
  }

  [[nodiscard]] size_t getAvailableWorkerCount(
      WorkerSchedulePolicy policy) const;

  void schedule(WorkerSchedulePolicy policy,
                const std::function<void(const topology::core*, bool)>&,
                size_t num_workers);
  void schedule(WorkerSchedulePolicy policy,
                const std::function<void(const topology::core*, bool)>&,
                size_t num_workers, bool reversed);

 private:
  size_t total_worker_count;
  size_t total_worker_per_numa;
  size_t ht_couple_size;

  std::map<core_id_t, bool> physical_thread_map;
  std::set<core_id_t> all_hyper_threads;
  std::set<core_id_t> all_physical_threads;

 private:
  ScheduleWorkers();
  ~ScheduleWorkers() = default;

  [[nodiscard]] static std::pair<std::vector<const topology::core*>,
                                 std::vector<const topology::core*>>
  segregatePhysicalAndHyperThreads(const std::vector<topology::core>&);
  void markPhysicalThreads();
};

}  // namespace scheduler

#endif  // PROTEUS_WORKER_SCHEDULE_POLICY_HPP
