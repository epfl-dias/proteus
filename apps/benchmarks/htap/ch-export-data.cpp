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
#include <unistd.h>

#include <cmath>
#include <iostream>
#include <string>

// HTAP
#include "htap-cli-flags.hpp"

// UTILS
#include "util/profiling.hpp"
#include "util/timing.hpp"

// OLAP
#include <common/olap-common.hpp>

#include "memory/memory-manager.hpp"
#include "queries/olap-sequence.hpp"
#include "storage/storage-manager.hpp"

// OLTP
#include "aeolus-plugin.hpp"
#include "oltp.hpp"
#include "tpcc_64.hpp"
#include "ycsb.hpp"

// Platform

#include "queries/ch/ch-queries.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

DEFINE_string(ch_export_dir, "/scratch1/export",
              "Export dir for ch-data-export");

void init(int argc, char *argv[]) {
  gflags::SetUsageMessage("Simple command line interface for proteus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;  // FIXME: the command line flags/defs seem to fail...
  google::InstallFailureSignalHandler();
  // set_trace_allocations(true);
}

int main(int argc, char *argv[]) {
  // assert(FLAGS_ch_export_dir.length() > 2);

  init(argc, argv);
  proteus::olap::init();

  OLTP oltp_engine;

  // OLTP
  uint oltp_data_partitions = 1;
  uint oltp_num_workers = topology::getInstance().getCoreCount() /
                          topology::getInstance().getCpuNumaNodes().size();

  g_num_partitions = oltp_data_partitions;

  LOG(INFO) << "[OLTP] CH Scale Factor: " << FLAGS_ch_scale_factor;

  oltp_engine.init(new bench::TPCC("TPCC", oltp_num_workers, oltp_num_workers,
                                   true, FLAGS_ch_scale_factor, 0, "", false),
                   oltp_num_workers, oltp_data_partitions,
                   FLAGS_ch_scale_factor);

  LOG(INFO) << "[OLTP] Initialization completed.";

  oltp_engine.print_storage_stats();
  oltp_engine.snapshot();

  LOG(INFO) << "Export Begin.";

  DataExporter_CH::exportAll(FLAGS_ch_export_dir);

  LOG(INFO) << "Export completed.";

  // OLAP
  StorageManager::unloadAll();
  MemoryManager::destroy();

  // OLTP
  oltp_engine.shutdown();

  return 0;
}
