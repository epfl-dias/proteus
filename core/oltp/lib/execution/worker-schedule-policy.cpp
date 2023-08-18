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

#include "oltp/execution/worker-schedule-policy.hpp"

#include <oltp/common/constants.hpp>
#include <ranges>

namespace scheduler {

std::ostream &operator<<(std::ostream &out, WorkerSchedulePolicy policy) {
  constexpr auto reversed =
      global_conf::reverse_partition_numa_mapping ? " (reversed)" : "";

  switch (policy) {
    case WorkerSchedulePolicy::PHYSICAL_ONLY:
      return out << "PHYSICAL_ONLY" << reversed;
    case WorkerSchedulePolicy::PHYSICAL_FIRST:
      return out << "PHYSICAL_FIRST" << reversed;
    case WorkerSchedulePolicy::SOCKET_FIRST_PHYSICAL_FIRST:
      return out << "SOCKET_FIRST_PHYSICAL_FIRST" << reversed;
    case WorkerSchedulePolicy::CORE_FIRST:
      return out << "CORE_FIRST" << reversed;
    case WorkerSchedulePolicy::HALF_SOCKET_PHYSICAL:
      return out << "HALF_SOCKET_PHYSICAL" << reversed;
    case WorkerSchedulePolicy::HALF_SOCKET:
      return out << "HALF_SOCKET" << reversed;
    case WorkerSchedulePolicy::INTERLEAVE_EVEN:
      return out << "INTERLEAVE_EVEN" << reversed;
    case WorkerSchedulePolicy::INTERLEAVE_ODD:
      return out << "INTERLEAVE_ODD" << reversed;

      // omit default case to trigger compiler warning for missing cases
  }
  return out << static_cast<std::uint16_t>(policy);
}

ScheduleWorkers::ScheduleWorkers() {
  this->total_worker_count = topology::getInstance().getCoreCount();
  this->total_worker_per_numa = topology::getInstance().getCoreCount() /
                                topology::getInstance().getCpuNumaNodeCount();

  this->ht_couple_size =
      topology::getInstance().getCores()[0].ht_pairs_id.size();
  if (this->ht_couple_size == 0) {
    this->ht_couple_size = 1;
  }

  this->markPhysicalThreads();
}

void ScheduleWorkers::markPhysicalThreads() {
  const auto &all_cores = topology::getInstance().getCores();

  std::set<core_id_t> hyper_thread_ids;

  for (const auto &exec_core : all_cores) {
    if (hyper_thread_ids.contains(exec_core.id)) {
      physical_thread_map[exec_core.id] = false;
    } else {
      physical_thread_map[exec_core.id] = true;
      all_physical_threads.emplace(exec_core.id);
      for (const auto &sibling : exec_core.ht_pairs_id) {
        hyper_thread_ids.emplace(sibling);
        all_hyper_threads.emplace(sibling);
      }
    }
  }
}

std::pair<std::vector<const topology::core *>,
          std::vector<const topology::core *>>
ScheduleWorkers::segregatePhysicalAndHyperThreads(
    const std::vector<topology::core> &cores) {
  std::vector<const topology::core *> physicalCores;
  std::vector<const topology::core *> hyperThreads;

  std::set<core_id_t> hyper_thread_ids;

  for (const auto &exec_core : cores) {
    if (hyper_thread_ids.contains(exec_core.id)) {
      hyperThreads.push_back(&exec_core);
      //      LOG(INFO) << "Hyper: " << exec_core.id;
    } else {
      //      LOG(INFO) << "Physical: " << exec_core.id;
      physicalCores.push_back(&exec_core);
      for (const auto &sibling : exec_core.ht_pairs_id) {
        //        LOG(INFO) << "\t Sibling HT: " << sibling;
        hyper_thread_ids.emplace(sibling);
      }
    }
  }
  return {physicalCores, hyperThreads};
}

size_t ScheduleWorkers::getAvailableWorkerCount(
    WorkerSchedulePolicy policy) const {
  switch (policy) {
    case WorkerSchedulePolicy::PHYSICAL_ONLY:
      return total_worker_count / ht_couple_size;
    case WorkerSchedulePolicy::PHYSICAL_FIRST:
    case WorkerSchedulePolicy::SOCKET_FIRST_PHYSICAL_FIRST:
    case WorkerSchedulePolicy::CORE_FIRST:
      return total_worker_count;

    case WorkerSchedulePolicy::HALF_SOCKET_PHYSICAL: {
      size_t total = 0;
      for (const auto &numa : topology::getInstance().getCpuNumaNodes()) {
        total += (numa.local_cores.size() / ht_couple_size) / 2;
      }
      return total;
    }
    case WorkerSchedulePolicy::HALF_SOCKET: {
      size_t total = 0;
      for (const auto &numa : topology::getInstance().getCpuNumaNodes()) {
        total += numa.local_cores.size() / 2;
      }
      return total;
    }
    case WorkerSchedulePolicy::INTERLEAVE_EVEN: {
      auto core_count = total_worker_count;
      if (core_count % 2 == 0) {
        return core_count / 2;
      } else {
        return (core_count) - ((core_count / 2) + 1);
      }
    }
    case WorkerSchedulePolicy::INTERLEAVE_ODD: {
      auto core_count = total_worker_count;
      if (core_count % 2 == 0) {
        return core_count / 2;
      } else {
        return ((core_count / 2) + 1);
      }
    }
    default:
      // omit default case to trigger compiler warning for missing cases
      assert(false && "missing switch case?");
  }
}

static auto inline filterCoresById(std::vector<const topology::core *> &threads,
                                   const std::vector<core_id_t> &core_ids) {
  std::vector<const topology::core *> physicalNuma;
  std::copy_if(threads.begin(), threads.end(), std::back_inserter(physicalNuma),
               [&](const topology::core *elem) {
                 return any_of(
                     core_ids.begin(), core_ids.end(),
                     [&](core_id_t core_id) { return elem->id == core_id; });
               });

  return physicalNuma;
}

void ScheduleWorkers::schedule(
    WorkerSchedulePolicy policy,
    const std::function<void(const topology::core *workerThread,
                             bool isPhysical)> &createWorker,
    size_t num_workers) {
  bool reversed = global_conf::reverse_partition_numa_mapping;
  this->schedule(policy, createWorker, num_workers, reversed);
}

void ScheduleWorkers::schedule(
    WorkerSchedulePolicy policy,
    const std::function<void(const topology::core *workerThread,
                             bool isPhysical)> &createWorker,
    size_t num_workers, bool reversed) {
  if (num_workers == 0) {
    num_workers = this->total_worker_count;
  }

  LOG(INFO) << "num_workers: " << num_workers;
  // reverse will start from the last NUMA socket, filling the way back in.

  LOG_IF(FATAL,
         num_workers > 0 && num_workers > (getAvailableWorkerCount(policy)))
      << "Requested more # of workers than available (available: "
      << getAvailableWorkerCount(policy) << ")";

  if (policy == PHYSICAL_ONLY) {
    // use physical threads only across NUMA
    size_t i = 0;
    auto allThreads =
        segregatePhysicalAndHyperThreads(topology::getInstance().getCores());
    auto physicalThreads = allThreads.first;

    if (reversed) {
      for (auto it = physicalThreads.rbegin();
           it != physicalThreads.rend() && i < num_workers; i++, it++) {
        createWorker(*it, physical_thread_map.at((*it)->id));
      }
    } else {
      for (auto it = physicalThreads.begin();
           it != physicalThreads.end() && i < num_workers; i++, it++) {
        createWorker(*it, physical_thread_map.at((*it)->id));
      }
    }
  } else if (policy == PHYSICAL_FIRST) {
    // first use physical, then HT across NUMA
    size_t i = 0;
    auto allThreads =
        segregatePhysicalAndHyperThreads(topology::getInstance().getCores());
    auto physicalThreads = allThreads.first;
    auto hyperThreads = allThreads.second;

    if (reversed) {
      for (auto it = physicalThreads.rbegin();
           it != physicalThreads.rend() && i < num_workers; i++, it++) {
        createWorker(*it, physical_thread_map.at((*it)->id));
      }
      for (auto it = hyperThreads.rbegin();
           it != hyperThreads.rend() && i < num_workers; i++, it++) {
        createWorker(*it, physical_thread_map.at((*it)->id));
      }

    } else {
      for (auto it = physicalThreads.begin();
           it != physicalThreads.end() && i < num_workers; i++, it++) {
        createWorker(*it, physical_thread_map.at((*it)->id));
      }
      for (auto it = hyperThreads.begin();
           it != hyperThreads.end() && i < num_workers; i++, it++) {
        createWorker(*it, physical_thread_map.at((*it)->id));
      }
    }
  } else if (policy == SOCKET_FIRST_PHYSICAL_FIRST) {
    // first physical, then HT, before next NUMA

    size_t i = 0;
    auto allThreads =
        segregatePhysicalAndHyperThreads(topology::getInstance().getCores());
    auto physicalThreads = allThreads.first;
    auto hyperThreads = allThreads.second;

    if (reversed) {
      for (auto numa = topology::getInstance().getCpuNumaNodes().rbegin();
           numa != topology::getInstance().getCpuNumaNodes().rend() &&
           i < num_workers;
           numa++) {
        std::vector<const topology::core *> physicalNuma =
            filterCoresById(physicalThreads, numa->local_cores);
        std::vector<const topology::core *> HyperNuma =
            filterCoresById(hyperThreads, numa->local_cores);

        for (auto it = physicalNuma.rbegin();
             it != physicalNuma.rend() && i < num_workers; i++, it++) {
          createWorker(*it, physical_thread_map.at((*it)->id));
        }
        for (auto it = HyperNuma.rbegin();
             it != HyperNuma.rend() && i < num_workers; i++, it++) {
          createWorker(*it, physical_thread_map.at((*it)->id));
        }
      }

    } else {
      for (const auto &numa : topology::getInstance().getCpuNumaNodes()) {
        std::vector<const topology::core *> physicalNuma =
            filterCoresById(physicalThreads, numa.local_cores);
        std::vector<const topology::core *> HyperNuma =
            filterCoresById(hyperThreads, numa.local_cores);

        for (auto it = physicalNuma.begin();
             it != physicalNuma.end() && i < num_workers; i++, it++) {
          createWorker(*it, physical_thread_map.at((*it)->id));
        }
        for (auto it = HyperNuma.begin();
             it != HyperNuma.end() && i < num_workers; i++, it++) {
          createWorker(*it, physical_thread_map.at((*it)->id));
        }

        if (i >= num_workers) break;
      }
    }

  } else if (policy == CORE_FIRST) {
    // physical + HT of same core before next core
    std::map<core_id_t, bool> done;
    const auto &all_cores = topology::getInstance().getCores();
    size_t i = 0;

    if (reversed) {
      for (auto it = all_cores.rbegin();
           it != all_cores.rend() && i < num_workers; it++) {
        if (done.contains(it->id) && done.at(it->id)) {
          continue;
        }

        createWorker(&(*it), physical_thread_map.at(it->id));
        done[it->id] = true;
        i++;

        for (const auto &ht_id : it->ht_pairs_id) {
          if (done.contains(ht_id) && done[ht_id]) {
            continue;
          }

          if (auto ht_it = std::find_if(
                  begin(all_cores), end(all_cores),
                  [ht_id](const topology::core &c) { return c.id == ht_id; });
              ht_it != std::end(all_cores)) {
            createWorker(&(*ht_it), physical_thread_map.at(ht_it->id));
            done[ht_id] = true;
            i++;
            if (i >= num_workers) break;
          }
        }
      }

    } else {
      for (auto it = all_cores.begin();
           it != all_cores.end() && i < num_workers; it++) {
        if (done.contains(it->id) && done.at(it->id)) {
          continue;
        }

        createWorker(&(*it), physical_thread_map.at(it->id));
        done[it->id] = true;
        i++;

        for (const auto &ht_id : it->ht_pairs_id) {
          if (done.contains(ht_id) && done[ht_id]) {
            continue;
          }

          if (auto ht_it = std::find_if(
                  begin(all_cores), end(all_cores),
                  [ht_id](const topology::core &c) { return c.id == ht_id; });
              ht_it != std::end(all_cores)) {
            createWorker(&(*ht_it), physical_thread_map.at(ht_it->id));
            done[ht_id] = true;
            i++;
            if (i >= num_workers) break;
          }
        }
      }
    }

  } else if (policy == HALF_SOCKET_PHYSICAL) {
    // half of one socket, and then half of 2nd socket
    //  -> if reversed, then second half.

    size_t i = 0;
    auto allThreads =
        segregatePhysicalAndHyperThreads(topology::getInstance().getCores());
    auto physicalThreads = allThreads.first;

    if (reversed) {
      for (auto numa = topology::getInstance().getCpuNumaNodes().rbegin();
           numa != topology::getInstance().getCpuNumaNodes().rend() &&
           i < num_workers;
           numa++) {
        std::vector<const topology::core *> physicalNuma =
            filterCoresById(physicalThreads, numa->local_cores);

        size_t max_p_in_socket = physicalNuma.size() / 2;
        size_t i_in_socket = 0;

        for (auto it = physicalNuma.rbegin();
             it != physicalNuma.rend() && i < num_workers &&
             i_in_socket < max_p_in_socket;
             i++, it++, i_in_socket++) {
          createWorker(*it, physical_thread_map.at((*it)->id));
        }
      }
    } else {
      for (auto numa = topology::getInstance().getCpuNumaNodes().begin();
           numa != topology::getInstance().getCpuNumaNodes().end() &&
           i < num_workers;
           numa++) {
        std::vector<const topology::core *> physicalNuma =
            filterCoresById(physicalThreads, numa->local_cores);

        size_t max_p_in_socket = physicalNuma.size() / 2;
        size_t i_in_socket = 0;

        for (auto it = physicalNuma.begin();
             it != physicalNuma.end() && i < num_workers &&
             i_in_socket < max_p_in_socket;
             i++, it++, i_in_socket++) {
          createWorker(*it, physical_thread_map.at((*it)->id));
        }
      }
    }

  } else if (policy == HALF_SOCKET) {
    // fill half socket with OLTP engine
    //  -> if reversed, then second half of both socket.
    // first physical, then HT, before next NUMA

    size_t i = 0;
    auto allThreads =
        segregatePhysicalAndHyperThreads(topology::getInstance().getCores());
    auto physicalThreads = allThreads.first;
    auto hyperThreads = allThreads.second;

    if (reversed) {
      for (auto numa = topology::getInstance().getCpuNumaNodes().rbegin();
           numa != topology::getInstance().getCpuNumaNodes().rend() &&
           i < num_workers;
           numa++) {
        std::vector<const topology::core *> physicalNuma =
            filterCoresById(physicalThreads, numa->local_cores);
        std::vector<const topology::core *> HyperNuma =
            filterCoresById(hyperThreads, numa->local_cores);

        size_t max_p_in_socket = physicalNuma.size() / 2;
        size_t max_h_in_socket = HyperNuma.size() / 2;
        size_t i_in_socket = 0;

        for (auto it = physicalNuma.rbegin();
             it != physicalNuma.rend() && i < num_workers &&
             i_in_socket < max_p_in_socket;
             i++, it++, i_in_socket++) {
          createWorker(*it, true);
        }

        i_in_socket = 0;
        for (auto it = HyperNuma.rbegin();
             it != HyperNuma.rend() && i < num_workers &&
             i_in_socket < max_h_in_socket;
             i++, it++, i_in_socket++) {
          createWorker(*it, false);
        }
      }
    } else {
      for (auto numa = topology::getInstance().getCpuNumaNodes().begin();
           numa != topology::getInstance().getCpuNumaNodes().end() &&
           i < num_workers;
           numa++) {
        std::vector<const topology::core *> physicalNuma =
            filterCoresById(physicalThreads, numa->local_cores);
        std::vector<const topology::core *> HyperNuma =
            filterCoresById(hyperThreads, numa->local_cores);

        size_t max_p_in_socket = physicalNuma.size() / 2;
        size_t max_h_in_socket = HyperNuma.size() / 2;
        size_t i_in_socket = 0;

        for (auto it = physicalNuma.begin();
             it != physicalNuma.end() && i < num_workers &&
             i_in_socket < max_p_in_socket;
             i++, it++, i_in_socket++) {
          createWorker(*it, true);
        }

        i_in_socket = 0;
        for (auto it = HyperNuma.begin();
             it != HyperNuma.end() && i < num_workers &&
             i_in_socket < max_h_in_socket;
             i++, it++, i_in_socket++) {
          createWorker(*it, false);
        }
      }
    }

  } else if (policy == INTERLEAVE_EVEN) {
    const auto &all_cores = topology::getInstance().getCores();
    size_t i = 0, j = 0;
    if (reversed) {
      for (auto it = all_cores.rbegin();
           it != all_cores.rend() && i < num_workers; j++, it++) {
        if (j % 2 == 0) {
          createWorker(&(*it), physical_thread_map.at(it->id));
        }
      }
    } else {
      for (auto it = all_cores.begin();
           it != all_cores.end() && i < num_workers; j++, it++) {
        if (j % 2 == 0) {
          createWorker(&(*it), physical_thread_map.at(it->id));
          i++;
        }
      }
    }
  } else if (policy == INTERLEAVE_ODD) {
    const auto &all_cores = topology::getInstance().getCores();
    size_t i = 0, j = 0;
    if (reversed) {
      for (auto it = all_cores.rbegin();
           it != all_cores.rend() && i < num_workers; j++, it++) {
        if (j % 2 != 0) {
          createWorker(&(*it), physical_thread_map.at(it->id));
          i++;
        }
      }
    } else {
      for (auto it = all_cores.begin();
           it != all_cores.end() && i < num_workers; j++, it++) {
        if (j % 2 != 0) {
          createWorker(&(*it), physical_thread_map.at(it->id));
          i++;
        }
      }
    }
  } else {
    LOG(FATAL) << "Unknown worker scheduling policy: " << policy;
  }
}

}  // namespace scheduler
