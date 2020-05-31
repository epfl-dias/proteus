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
#include <gflags/gflags.h>

#include <iostream>
#include <memory/memory-manager.hpp>
#include <olap/common/olap-common.hpp>
#include <olap/util/context.hpp>
#include <topology/topology.hpp>
#include <util/glog.hpp>

static bool validatePercentage(const char *flagname, double value) {
  if (value >= 0.0 && value <= 1.0) return true;
  std::cerr << "Invalid value for --" << flagname << ": " << value << std::endl;
  return false;
}

DEFINE_bool(query_topology, false, "Print the system topology and exit");
DEFINE_bool(trace_allocations, false,
            "Trace memory allocation and leaks (requires a build with "
            "undefined NDEBUG)");
DEFINE_double(gpu_buffers, 0.2,
              "Percentage (0.0-1.0) of GPU memory to dedicate for buffer "
              "management (per GPU)");
DEFINE_validator(gpu_buffers, &validatePercentage);
DEFINE_double(cpu_buffers, 0.1,
              "Percentage (0.0-1.0) of CPU memory to dedicate for buffer "
              "management (per CPU)");
DEFINE_validator(cpu_buffers, &validatePercentage);
DEFINE_int64(
    log_buffer_usage, 0,
    "Periodically print buffer usage in stderr, value is interval (in ms) "
    "between prints, 0 to disable");
DEFINE_bool(primary, false, "Make this instance a primary node");
DEFINE_bool(secondary, false, "Connect to a primary node");
DEFINE_bool(ipv4, false, "Use IPv4");
DEFINE_int32(port, 12345,
             "Used in conjuction with --secondary to specify the listening "
             "port of the primary");
DEFINE_string(url, "localhost",
              "Used in conjuction with --secondary to specify the address of "
              "the primary");
DEFINE_int32(repeat, 1, "# repetitions of default query");
DEFINE_bool(print_generated_code, false,
            "Print generated code into files (use only for debugging as it "
            "will slow down excecution significnatly)");

static bool validatePort(const char *flag, int32_t value) {
  if (value > 0 && value < 0x8000) return true;  // max port value: 32768
  std::cout << "Invalid value for --" << flag << ": " << value << std::endl;
  return false;
}

DEFINE_validator(port, &validatePort);

namespace proteus::from_cli {
proteus::olap olap() {
  srand(time(nullptr));

  google::InstallFailureSignalHandler();

  if (FLAGS_query_topology) {
    topology::init();
    std::cout << topology::getInstance() << std::endl;
    exit(0);
  }

  set_trace_allocations(FLAGS_trace_allocations);
  print_generated_code = FLAGS_print_generated_code;

  return proteus::olap{static_cast<float>(FLAGS_gpu_buffers),
                       static_cast<float>(FLAGS_cpu_buffers),
                       static_cast<size_t>(FLAGS_log_buffer_usage)};
}

proteus::olap olap(const std::string &usage, int *argc, char ***argv) {
  gflags::SetUsageMessage(usage);
  gflags::ParseCommandLineFlags(argc, argv, true);

  google::InitGoogleLogging((*argv)[0]);
  FLAGS_logtostderr = true;  // FIXME: the command line flags/defs seem to fail

  return olap();
}
}  // namespace proteus::from_cli
