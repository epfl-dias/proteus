/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#include "common/common.hpp"

#include <thread>

#include "communication/comm-manager.hpp"
#include "memory/memory-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/jit/pipeline.hpp"

namespace proteus {

void thread_warm_up() {}

void init(float gpu_mem_pool_percentage, float cpu_mem_pool_percentage,
          size_t log_buffers) {
  topology::init();

  // Initialize Google's logging library.
  LOG(INFO) << "Starting up server...";

  // Force initialization of communcation manager by getting the instance
  LOG(INFO) << "Initializing communication manager...";
  communication::CommManager::getInstance();

  LOG(INFO) << "Warming up GPUs...";
  for (const auto &gpu : topology::getInstance().getGpus()) {
    set_exec_location_on_scope d{gpu};
    gpu_run(cudaFree(nullptr));
  }

  gpu_run(cudaFree(nullptr));

  // gpu_run(cudaDeviceSetLimit(cudaLimitStackSize, 40960));

  LOG(INFO) << "Warming up threads...";

  std::vector<std::thread> thrds;
  for (int i = 0; i < 1024; ++i) thrds.emplace_back(thread_warm_up);
  for (auto &t : thrds) t.join();

  // srand(time(0));

  LOG(INFO) << "Initializing codegen...";

  PipelineGen::init();

  LOG(INFO) << "Initializing memory manager...";
  MemoryManager::init(gpu_mem_pool_percentage, cpu_mem_pool_percentage,
                      log_buffers);

  // Make affinity deterministic
  auto &topo = topology::getInstance();
  if (topo.getGpuCount() > 0) {
    exec_location{topo.getGpus()[0]}.activate();
  } else {
    exec_location{topo.getCpuNumaNodes()[0]}.activate();
  }
}

}  // namespace proteus
