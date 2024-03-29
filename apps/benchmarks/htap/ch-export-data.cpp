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

#include <cmath>
#include <iostream>

// HTAP
#include "htap-cli-flags.hpp"

// UTILS
#include <platform/util/timing.hpp>

// OLAP
#include <cli-flags.hpp>
#include <olap/common/olap-common.hpp>

// OLTP
#include "oltp.hpp"
#include "tpcc/tpcc_64.hpp"
#include "ycsb.hpp"

// Platform

#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

#include "queries/ch/ch-queries.hpp"

DEFINE_string(ch_export_dir, "/scratch1/export",
              "Export dir for ch-data-export");

int main(int argc, char *argv[]) {
  auto olap = proteus::from_cli::olap("ch exporter", &argc, &argv);
  // assert(FLAGS_ch_export_dir.length() > 2);

  OLTP oltp_engine;

  // OLTP
  uint oltp_data_partitions = 1;
  uint oltp_num_workers = topology::getInstance().getCoreCount() /
                          topology::getInstance().getCpuNumaNodes().size();

  g_num_partitions = oltp_data_partitions;

  LOG(INFO) << "[OLTP] CH Scale Factor: " << FLAGS_ch_scale_factor;

  oltp_engine.init(new bench::TPCC("TPCC", oltp_num_workers, oltp_num_workers,
                                   {}, FLAGS_ch_scale_factor, 0, "", false),
                   oltp_num_workers, oltp_data_partitions,
                   FLAGS_ch_scale_factor);

  LOG(INFO) << "[OLTP] Initialization completed.";

  oltp_engine.print_storage_stats();
  oltp_engine.snapshot();

  LOG(INFO) << "Export Begin.";

  DataExporter_CH::exportAll(FLAGS_ch_export_dir);

  LOG(INFO) << "Export completed.";

  // OLTP
  oltp_engine.shutdown();

  return 0;
}
