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

#include <fstream>
#include <magic_enum.hpp>
#include <platform/common/common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/rdtsc.hpp>

double diff(struct timespec st, struct timespec end) {
  struct timespec tmp;

  if ((end.tv_nsec - st.tv_nsec) < 0) {
    tmp.tv_sec = end.tv_sec - st.tv_sec - 1;
    tmp.tv_nsec = 1e9 + end.tv_nsec - st.tv_nsec;
  } else {
    tmp.tv_sec = end.tv_sec - st.tv_sec;
    tmp.tv_nsec = end.tv_nsec - st.tv_nsec;
  }

  return tmp.tv_sec + tmp.tv_nsec * 1e-9;
}

std::ostream &operator<<(std::ostream &out, const bytes &b) {
  const char *units[]{"B", "KB", "MB", "GB", "TB", "ZB"};
  constexpr size_t max_i = sizeof(units) / sizeof(units[0]);
  size_t bs = b.b * 10;

  size_t i = 0;
  while (bs >= 10240 && i < max_i) {
    bs /= 1024;
    ++i;
  }

  out << (bs / 10.0) << units[i];
  return out;
}

namespace std {
std::string to_string(const bytes &b) {
  std::stringstream ss;
  ss << b;
  return ss.str();
}
};  // namespace std

namespace proteus {

void thread_warm_up() {}

class platform::impl {
 public:
  impl(float gpu_mem_pool_percentage, float cpu_mem_pool_percentage,
       size_t log_buffers) {
    topology::init();

    // Initialize Google's logging library.
    LOG(INFO) << "Starting up server...";

    LOG(INFO) << "Warming up GPUs...";
    for (const auto &gpu : topology::getInstance().getGpus()) {
      set_exec_location_on_scope d{gpu};
      gpu_run(cudaFree(nullptr));
    }

    gpu_run(cudaFree(nullptr));

    // gpu_run(cudaDeviceSetLimit(cudaLimitStackSize, 40960));

    LOG(INFO) << "Warming up threads...";

    //    std::vector<std::thread> thrds;
    //    thrds.reserve(32);
    //    for (int i = 0; i < 32; ++i) thrds.emplace_back(thread_warm_up);
    //    for (auto &t : thrds) t.join();

    // srand(time(0));

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

  ~impl() {
    LOG(INFO) << "Shutting down...";

    //    Unloading files on platform destruction is a little tricky, now that
    //    we have moved storage into its own lib which depends on platform
    //    LOG(INFO) << "Unloading files...";
    //    StorageManager::getInstance().unloadAll();

    LOG(INFO) << "Shuting down memory manager...";
    MemoryManager::destroy();

    LOG(INFO) << "Shut down finished";
  }
};

platform::platform(float gpu_mem_pool_percentage, float cpu_mem_pool_percentage,
                   size_t log_buffers)
    : p_impl(std::make_unique<impl>(gpu_mem_pool_percentage,
                                    cpu_mem_pool_percentage, log_buffers)) {}

platform::~platform() = default;

}  // namespace proteus
