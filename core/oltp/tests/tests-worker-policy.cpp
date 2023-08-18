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

#include <platform/util/timing.hpp>

#include "gtest/gtest.h"
#include "oltp/execution/worker-schedule-policy.hpp"
#include "test-utils.hpp"

::testing::Environment* const pools_env =
    ::testing::AddGlobalTestEnvironment(new TestEnvironment);

void testPolicy(scheduler::WorkerSchedulePolicy policy,
                const std::string& policy_string, size_t total_workers,
                bool reverse = false) {
  size_t deployed_worker = 0;
  LOG(INFO) << "-============ " << policy_string
            << " | total_worker: " << total_workers
            << " | reverse: " << reverse;
  scheduler::ScheduleWorkers::getInstance().schedule(
      policy,
      [&](const topology::core* workerThread, bool isPhysical) {
        deployed_worker++;
        LOG(INFO) << "Worker-" << deployed_worker << " | Core-"
                  << workerThread->id << "\t\t("
                  << (isPhysical ? "Physical" : "HyperThread")
                  << ") | CPU-ID: " << workerThread->local_cpu_id;
      },
      total_workers, reverse);

  LOG_IF(FATAL, deployed_worker != total_workers)
      << " deployed_worker != total_workers : " << deployed_worker
      << " != " << total_workers;
}

TEST(OltpWorkerScheduler, testAllPolicies) {
  auto total_cores = topology::getInstance().getCoreCount();

  testPolicy(scheduler::PHYSICAL_ONLY, "PHYSICAL_ONLY", total_cores / 2);
  testPolicy(scheduler::PHYSICAL_ONLY, "PHYSICAL_ONLY", total_cores / 2, true);

  testPolicy(scheduler::PHYSICAL_FIRST, "PHYSICAL_FIRST", total_cores);
  testPolicy(scheduler::PHYSICAL_FIRST, "PHYSICAL_FIRST", total_cores, true);

  testPolicy(scheduler::SOCKET_FIRST_PHYSICAL_FIRST,
             "SOCKET_FIRST_PHYSICAL_FIRST", total_cores);
  testPolicy(scheduler::SOCKET_FIRST_PHYSICAL_FIRST,
             "SOCKET_FIRST_PHYSICAL_FIRST", total_cores, true);

  testPolicy(scheduler::CORE_FIRST, "CORE_FIRST", total_cores);
  testPolicy(scheduler::CORE_FIRST, "CORE_FIRST", total_cores, true);

  testPolicy(scheduler::HALF_SOCKET_PHYSICAL, "HALF_SOCKET_PHYSICAL",
             total_cores / 4);
  testPolicy(scheduler::HALF_SOCKET_PHYSICAL, "HALF_SOCKET_PHYSICAL",
             total_cores / 4, true);

  testPolicy(scheduler::HALF_SOCKET, "HALF_SOCKET", total_cores / 2);
  testPolicy(scheduler::HALF_SOCKET, "HALF_SOCKET", total_cores / 2, true);

  testPolicy(scheduler::INTERLEAVE_EVEN, "INTERLEAVE_EVEN", total_cores / 2);
  testPolicy(scheduler::INTERLEAVE_EVEN, "INTERLEAVE_EVEN", total_cores / 2,
             true);

  testPolicy(scheduler::INTERLEAVE_ODD, "INTERLEAVE_ODD", total_cores / 2);
  testPolicy(scheduler::INTERLEAVE_ODD, "INTERLEAVE_ODD", total_cores / 2,
             true);
}
