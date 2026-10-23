/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#include <cstdlib>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/profiling.hpp>

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#define __itt_resume() ((void)0)
#define __itt_pause() ((void)0)
#endif

namespace profiling {
void resume() {
  for (const auto& gpu : topology::getInstance().getGpus()) {
    set_exec_location_on_scope d{gpu};
    // Defined by gpu-common.hpp if NCUDA is defined
    gpu_run(cudaProfilerStart());
  }
  __itt_resume();

  /**
   * see man perf record and the --control= option
   * We could be more efficient by saving the file descriptors in a global or
   * making this a class/singleton, but this is a simple way to do it and its
   * not generally performance critical
   */
  int perf_ctl_fd = -1;
  int perf_ctl_ack_fd = -1;
  char* PERF_CTL_FD = getenv("PERF_CTL_FD");
  char* PERF_CTL_ACK_FD = getenv("PERF_CTL_ACK_FD");

  if (PERF_CTL_FD != nullptr) {
    perf_ctl_fd = atoi(PERF_CTL_FD);
  } else {
    LOG_FIRST_N(WARNING, 1)
        << "PERF_CTL_FD envvar not set. Cannot programmatically "
           "control perf collection";
    return;
  }

  if (PERF_CTL_ACK_FD != nullptr) {
    perf_ctl_ack_fd = atoi(PERF_CTL_ACK_FD);
  } else {
    LOG_FIRST_N(WARNING, 1)
        << "PERF_CTL_FD set but PERF_CTL_ACK_FD envvar is not set. Cannot "
           "confirm that perf collection control commands are received";
  }

  // Start the performance counter
  PCHECK(write(perf_ctl_fd, "enable\n", 8) == 8)
      << "failed to write to perf control fd";

  if (perf_ctl_ack_fd != -1) {
    // Wait for the ack from the perf control process
    char ack[5];
    PCHECK(read(perf_ctl_ack_fd, ack, 5) == 5)
        << "failed to read from perf control ack fd";
    CHECK_EQ(strcmp(ack, "ack\n"), 0);
  }
}

void pause() {
  for (const auto& gpu : topology::getInstance().getGpus()) {
    set_device_on_scope d{gpu};
    // Defined by gpu-common.hpp if NCUDA is defined
    gpu_run(cudaProfilerStop());
  }
  __itt_pause();

  int perf_ctl_fd = -1;
  int perf_ctl_ack_fd = -1;
  char* PERF_CTL_FD = getenv("PERF_CTL_FD");
  char* PERF_CTL_ACK_FD = getenv("PERF_CTL_ACK_FD");

  if (PERF_CTL_FD != nullptr) {
    perf_ctl_fd = std::stoi(PERF_CTL_FD);
  } else {
    LOG_FIRST_N(WARNING, 1)
        << "PERF_CTL_FD envvar not set. Cannot programmatically "
           "control perf collection";
    return;
  }

  if (PERF_CTL_ACK_FD != nullptr) {
    perf_ctl_ack_fd = std::stoi(PERF_CTL_ACK_FD);
  } else {
    LOG_FIRST_N(WARNING, 1)
        << "PERF_CTL_FD set but PERF_CTL_ACK_FD envvar is not set. Cannot "
           "confirm that perf collection control commands are received";
  }

  // Start the performance counter
  PCHECK(write(perf_ctl_fd, "disable\n", 9) == 9)
      << "failed to write to perf control fd";

  if (perf_ctl_ack_fd != -1) {
    // Wait for the ack from the perf control process
    char ack[5];
    PCHECK(read(perf_ctl_ack_fd, ack, 5) == 5)
        << "failed to read from perf control ack fd";
    CHECK_EQ(strcmp(ack, "ack\n"), 0);
  }
}

}  // namespace profiling
