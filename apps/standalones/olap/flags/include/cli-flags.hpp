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

#include <common/olap-common.hpp>
#include <memory>

DECLARE_bool(query_topology);
DECLARE_bool(trace_allocations);
DECLARE_double(gpu_buffers);
DECLARE_double(cpu_buffers);
DECLARE_int64(log_buffer_usage);
DECLARE_bool(primary);
DECLARE_bool(secondary);
DECLARE_bool(ipv4);
DECLARE_int32(port);
DECLARE_string(url);
DECLARE_int32(repeat);
DECLARE_bool(print_generated_code);

namespace proteus::from_cli {
proteus::olap olap(const std::string &usage, int *argc, char ***argv);
proteus::olap olap();
}  // namespace proteus::from_cli
