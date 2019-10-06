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

#include "util/glog.hpp"

static bool validatePercentage(const char *flagname, double value) {
  if (value >= 0.0 && value <= 1.0) return true;
  std::cerr << "Invalid value for --" << flagname << ": " << value << std::endl;
  return false;
}

DEFINE_bool(query_topology, false, "Print the system topology and exit");
DEFINE_bool(trace_allocations, false,
            "Trace memory allocation and leaks (requires a build with "
            "undefined NDEBUG)");
DEFINE_double(gpu_buffers, 0.1,
              "Percentage (0.0-1.0) of GPU memory to dedicate for buffer "
              "management (per GPU)");
DEFINE_validator(gpu_buffers, &validatePercentage);
DEFINE_double(cpu_buffers, 0.1,
              "Percentage (0.0-1.0) of CPU memory to dedicate for buffer "
              "management (per CPU)");
DEFINE_validator(cpu_buffers, &validatePercentage);
DEFINE_bool(log_buffer_usage, false,
            "Periodically print buffer usage in stderr");
DEFINE_bool(primary, false, "Make this instance a primary node");
DEFINE_bool(secondary, false, "Connect to a primary node");
DEFINE_bool(ipv4, false, "Use IPv4");
DEFINE_int32(port, 12345,
             "Used in conjuction with --secondary to specify the listening "
             "port of the primary");
DEFINE_string(url, "localhost",
              "Used in conjuction with --secondary to specify the address of "
              "the primary");
DEFINE_int32(repeat, 5, "# repetitions of default query");

static bool validatePort(const char *flag, int32_t value) {
  if (value > 0 && value < 0x8000) return true;  // max port value: 32768
  std::cout << "Invalid value for --" << flag << ": " << value << std::endl;
  return false;
}

DEFINE_validator(port, &validatePort);
