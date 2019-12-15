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

#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"

void init(int argc, char *argv[]) {
  gflags::SetUsageMessage("Simple command line interface for proteus");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;  // FIXME: the command line flags/defs seem to fail...
  google::InstallFailureSignalHandler();
}

int main(int argc, char *argv[]) {
  init(argc, argv);
  proteus::olap::init();
  OLTP oltp_engine;

  LOG(INFO) << "HTAP Initializing";
  uint OLAP_socket = 0;
  uint OLTP_socket = 1;

  LOG(INFO) << "OLTP Initializing";

  // OLTP
  uint oltp_data_partitions = 1;
  uint oltp_num_workers = 1;

  if (FLAGS_num_oltp_clients == 0) {
    oltp_num_workers = topology::getInstance().getCoreCount();
  } else {
    oltp_num_workers = FLAGS_num_oltp_clients;
  }
  if (oltp_num_workers >
      topology::getInstance().getCpuNumaNodes()[0].local_cores.size()) {
    uint core_per_socket =
        topology::getInstance().getCpuNumaNodes()[0].local_cores.size();

    oltp_data_partitions =
        (int)ceil((double)oltp_num_workers / (double)core_per_socket);

    assert(oltp_data_partitions <=
           topology::getInstance().getCpuNumaNodeCount());
    g_num_partitions = oltp_data_partitions;
  }

  LOG(INFO) << "[OLTP] data-partitions: " << oltp_data_partitions;
  LOG(INFO) << "[OLTP] Txn workers: " << oltp_num_workers;
  LOG(INFO) << "[OLTP] CH Scale Factor: " << FLAGS_ch_scale_factor;

  oltp_engine.init(new bench::TPCC("TPCC", oltp_num_workers, oltp_num_workers,
                                   true, FLAGS_ch_scale_factor, 0, "", false),
                   oltp_num_workers, oltp_data_partitions,
                   FLAGS_ch_scale_factor);
  LOG(INFO) << "[OLTP] Initialization completed.";

  oltp_engine.print_storage_stats();

  if (!global_conf::reverse_partition_numa_mapping) {
    OLAP_socket = topology::getInstance().getCpuNumaNodeCount() - 1;
    OLTP_socket = 0;
  } else {
    OLTP_socket = topology::getInstance().getCpuNumaNodeCount() - 1;
    OLAP_socket = 0;
  }

  // OLAP
  const auto &topo = topology::getInstance();
  const auto &nodes = topo.getCpuNumaNodes();
  exec_location{nodes[OLAP_socket]}.activate();

  oltp_engine.snapshot();
  if (FLAGS_etl) {
    oltp_engine.etl(OLAP_socket);
  }

  int client_id = 1;

  std::vector<topology::numanode *> oltp_nodes;
  std::vector<topology::numanode *> olap_nodes;

  // oltp_nodes.emplace_back((topology::numanode *)&nodes[OLTP_socket]);
  // olap_nodes.emplace_back((topology::numanode *)&nodes[OLAP_socket]);

  if (FLAGS_gpu_olap) {
    for (const auto &n : nodes) {
      oltp_nodes.emplace_back((topology::numanode *)&n);
    }

    for (const auto &gpu_n : topo.getGpus()) {
      olap_nodes.emplace_back((topology::numanode *)&gpu_n);
    }
  } else {
    // assert(false && "set nodes for cpu-only mode");
    // for (const auto &n : nodes) {
    //   oltp_nodes.emplace_back((topology::numanode *)&n);
    // }
    // for (const auto &n : nodes) {
    //   olap_nodes.emplace_back((topology::numanode *)&n);
    // }

    oltp_nodes.emplace_back((topology::numanode *)&nodes[OLTP_socket]);
    olap_nodes.emplace_back((topology::numanode *)&nodes[OLAP_socket]);
  }

  OLAPSequence olap_queries(
      OLAPSequence::wrapper_t<AeolusRemotePlugin>{}, 1, olap_nodes, oltp_nodes,
      (FLAGS_gpu_olap ? DeviceType::GPU : DeviceType::CPU));
  // nodes[OLAP_socket], nodes[OLTP_socket]);

  profiling::resume();

  if (FLAGS_run_oltp) {
    oltp_engine.run();
    usleep(2000000);  // stabilize
    oltp_engine.print_differential_stats();
  }

  if (FLAGS_elastic) {
    if (FLAGS_trade_core) {
      oltp_engine.migrate_worker(FLAGS_elastic);
    } else {
      oltp_engine.scale_down(FLAGS_elastic);
    }
  }

  usleep(2000000);  // stabilize

  exec_location{nodes[OLAP_socket]}.activate();

  for (int i = 0; i < FLAGS_num_olap_clients; i++) {
    oltp_engine.print_differential_stats();

    // oltp_engine.snapshot();
    // if (FLAGS_etl) oltp_engine.etl(OLAP_socket);

    // usleep(1000000);  // stabilize

    // oltp_engine.print_differential_stats();

    olap_queries.run(FLAGS_num_olap_repeat);
  }

  LOG(INFO) << "OLAP sequence completed.";

  LOG(INFO) << olap_queries;

  oltp_engine.print_differential_stats();
  oltp_engine.print_global_stats();

  if (!FLAGS_run_oltp) {
    // FIXME: hack because it needs to run before it can be stopped
    oltp_engine.run();
  }

  LOG(INFO) << "Shutdown initiated";

  // OLAP
  StorageManager::unloadAll();
  MemoryManager::destroy();

  // OLTP
  oltp_engine.shutdown();

  return 0;

  // FLAGS_num_olap_repeat =
  //(NUM_OLAP_REPEAT / FLAGS_num_olap_clients);  // warmmup
}
