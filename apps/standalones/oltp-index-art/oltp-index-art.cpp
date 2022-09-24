/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#include <gflags/gflags.h>
#include <unistd.h>

#include <bitset>
#include <cli-flags.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>
#include <tuple>

// ART Library
#include "oltp/index/ART/OptimisticLockCoupling/art.hpp"
#include "oltp/index/ART/art.hpp"

// OLTP Engine
#include "oltp/common/oltp-cli-flags.hpp"
#include "oltp/common/utils.hpp"
#include "oltp/execution/worker.hpp"
#include "oltp/interface/bench.hpp"
#include "oltp/storage/table.hpp"
#include "oltp/transaction/transaction_manager.hpp"

// Platform Includes
#include "platform/common/common.hpp"
#include "platform/topology/affinity_manager.hpp"
#include "platform/topology/topology.hpp"
#include "platform/util/profiling.hpp"

int main(int argc, char** argv) {
  auto olap = proteus::from_cli::olap(
      "Simple command line interface for aeolus", &argc, &argv);

  const auto& topo = topology::getInstance();
  const auto& nodes = topo.getCpuNumaNodes();
  //  set_exec_location_on_scope d{nodes[0]};

  set_exec_location_on_scope d{topo.getCores()[2]};

  std::vector<std::string> keys = {
      "aunn", "periklis", "natassa", "stella", "dias", "and", "ant",
      "any",  "are",      "art",     "bar",    "baz",  "foo"};

  std::vector<std::string> keys_mit_license = {"above",
                                               "ACTION",
                                               "all",
                                               "AN",
                                               "and",
                                               "any",
                                               "ARISING",
                                               "AS",
                                               "associated",
                                               "AUTHORS",
                                               "be",
                                               "BUT",
                                               "charge",
                                               "CLAIM",
                                               "conditions",
                                               "CONNECTION",
                                               "CONTRACT",
                                               "copies",
                                               "copy",
                                               "copyright",
                                               "DAMAGES",
                                               "deal",
                                               "DEALINGS",
                                               "distribute",
                                               "documentation",
                                               "EVENT",
                                               "EXPRESS",
                                               "files",
                                               "FITNESS",
                                               "following",
                                               "FOR",
                                               "free",
                                               "FROM",
                                               "furnished",
                                               "granted",
                                               "hereby",
                                               "HOLDERS",
                                               "IMPLIED",
                                               "included",
                                               "including",
                                               "KIND",
                                               "LIABILITY",
                                               "LIABLE",
                                               "limitation",
                                               "LIMITED",
                                               "MERCHANTABILITY",
                                               "merge",
                                               "modify",
                                               "NONINFRINGEMENT",
                                               "NOT",
                                               "notice",
                                               "obtaining",
                                               "OTHER",
                                               "OTHERWISE",
                                               "OUT",
                                               "PARTICULAR",
                                               "Permission",
                                               "permit",
                                               "person",
                                               "persons",
                                               "portions",
                                               "PROVIDED",
                                               "publish",
                                               "PURPOSE",
                                               "restriction",
                                               "rights",
                                               "sell",
                                               "shall",
                                               "software",
                                               "subject",
                                               "sublicense",
                                               "substantial",
                                               "the",
                                               "this",
                                               "TORT",
                                               "use",
                                               "WARRANTIES",
                                               "WARRANTY",
                                               "WHETHER",
                                               "whom",
                                               "WITH",
                                               "without"};

  // test for integer keys
  LOG(INFO) << "---------- TEST FOR INTEGER KEYS --------";
  std::vector<uint32_t> cases_int(100);
  std::iota(cases_int.begin(), cases_int.end(), 0);

  uint64_t n = 50;

  // Seed the random number generator
  std::random_device rd;
  std::mt19937 gen(rd());

  // Create a distribution object
  std::uniform_int_distribution<> dist(0, 255);

  std::vector<uint64_t> v(n);
  for (auto& x : v) {
    x = dist(gen);
  }

  vector<uint64_t> cases(n);
  iota(cases.begin(), cases.end(), 0);
  unsigned seed = 0;
  std::shuffle(cases.begin(), cases.end(), std::default_random_engine(seed));

  // memory warm-up sanity testing
  auto* x = static_cast<size_t*>(MemoryManager::mallocPinned(2_G));
  for (size_t i = 0; i < 2_G / sizeof(size_t); i++) {
    x[i] = i;
  }

  string insertName = "Insert " + std::to_string(n) + " Uint64 in ART: ";
  art_olc::ART<uint64_t, uint64_t> art(false);
  MemoryManager::freePinned(x);

  LOG(INFO) << "----- MULTI_THREAD TEST ------";
  profiling::resume();
  int thread_size = 1;
  n = n - n % thread_size;
  uint64_t task_size = n / thread_size;

  LOG(INFO) << n % thread_size;
  assert(n % thread_size == 0 && "should be divisible");

  for (uint64_t i = 0; i < n; i++) {
    art.insert(i, i);
  }
  profiling::resume();
  while (true) {
    time_block t(insertName);
    std::vector<proteus::thread> loaders;
    // assign tasks
    LOG(INFO) << "-------- Assign Start -------";
    for (int i = 0; i < thread_size; i++) {
      loaders.emplace_back([&art, i, task_size, &cases]() {
        // pin the thread to specific core.
        set_exec_location_on_scope ex{topology::getInstance().getCores()[i]};

        LOG(INFO) << "Thread-" << i << ": assigned from: " << (task_size * i)
                  << " to " << ((i + 1) * task_size);

        {
          time_block t1("Thread-" + std::to_string(i) + ": ");
          for (auto task_id = task_size * i; task_id < (i + 1) * task_size;
               task_id++) {
            //            LOG(INFO) << "ToInsert: " << cases[task_id] << " | "
            //            << task_id;
            art.insert(cases[task_id], cases[task_id]);
            //            art.find(cases[task_id]);
          }
        }
      });
    }

    for (auto& th : loaders) {
      th.join();
    }
    break;
  }
  profiling::pause();

  for (uint64_t i = 0; i < n; i++) {
    if (art.find(cases[i]) != cases[i]) LOG(INFO) << "ERROR: " << i;
  }

  //  profiling::resume();
  //  while (true) {
  //    time_block t(insertName);
  //    //    for (uint64_t i : cases) {
  //    for (uint64_t i = 0; i < n; i++) {
  //      art.insert(i, i);
  //    }
  //    break;
  //  }
  //  profiling::pause();
  //
  //  string findName = "Find " + std::to_string(n) + " Uint64 in ART: ";
  //  profiling::resume();
  //
  //  while (true) {
  //    time_block t(findName);
  //    for (uint64_t& i : cases) {
  //      art.find(i);
  //    }
  //    break;
  //  }
  //  profiling::pause();
}
