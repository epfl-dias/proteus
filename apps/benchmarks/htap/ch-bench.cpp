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
#include <platform/util/profiling.hpp>
#include <platform/util/timing.hpp>

// OLAP
#include <cli-flags.hpp>
#include <olap/common/olap-common.hpp>
#include <platform/memory/memory-manager.hpp>
#include <storage/storage-manager.hpp>

#include "queries/olap-sequence.hpp"

// OLTP
#include "aeolus-plugin.hpp"
#include "oltp.hpp"
#include "tpcc/tpcc_64.hpp"
#include "ycsb.hpp"

// Platform

#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>

int main(int argc, char *argv[]) {
  auto olap = proteus::from_cli::olap(
      "Simple command line interface for ch-bench", &argc, &argv);

  OLTP oltp_engine;

  LOG(INFO) << "HTAP Initializing";
  uint OLAP_socket = 0;
  uint OLTP_socket = 1;

  LOG(INFO) << "OLTP Initializing";

  // OLTP
  uint oltp_data_partitions = 1;
  uint oltp_num_workers = 1;

  if (FLAGS_num_oltp_clients == 0) {
    if (FLAGS_gpu_olap)
      oltp_num_workers = topology::getInstance().getCoreCount();
    else
      oltp_num_workers = topology::getInstance().getCoreCount() /
                         topology::getInstance().getCpuNumaNodes().size();
  } else {
    oltp_num_workers = FLAGS_num_oltp_clients;
  }

  assert(FLAGS_oltp_elastic_threshold < oltp_num_workers);

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

  bench::Benchmark *bench =
      new bench::TPCC("TPCC", oltp_num_workers, oltp_num_workers, true, {},
                      FLAGS_ch_scale_factor, 0, "", true);
  // if (FLAGS_htap_mode.compare("COLOC") == 0) {
  //   g_num_partitions = 2;
  //   oltp_data_partitions = 2;
  //   oltp_engine.init(bench, oltp_num_workers, oltp_data_partitions,
  //                    FLAGS_ch_scale_factor, true);

  // } else {
  //   oltp_engine.init(bench, oltp_num_workers, oltp_data_partitions,
  //                    FLAGS_ch_scale_factor);
  // }

  oltp_engine.init(bench, oltp_num_workers, oltp_data_partitions,
                   FLAGS_ch_scale_factor);

  LOG(INFO) << "[OLTP] data-partitions: " << oltp_data_partitions;
  LOG(INFO) << "[OLTP] Txn workers: " << oltp_num_workers;
  LOG(INFO) << "[OLTP] CH Scale Factor: " << FLAGS_ch_scale_factor;
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

  std::vector<topology::numanode *> oltp_nodes;
  std::vector<topology::numanode *> olap_nodes;

  if (FLAGS_gpu_olap) {
    for (const auto &n : nodes) {
      oltp_nodes.emplace_back((topology::numanode *)&n);
    }

    for (const auto &gpu_n : topo.getGpus()) {
      olap_nodes.emplace_back((topology::numanode *)&gpu_n);
    }
  } else {
    oltp_nodes.emplace_back((topology::numanode *)&nodes[OLTP_socket]);
    olap_nodes.emplace_back((topology::numanode *)&nodes[OLAP_socket]);
  }

  SchedulingPolicy::ScheduleMode schedule_policy =
      SchedulingPolicy::S2_ISOLATED;

  if (FLAGS_htap_mode.compare("ADAPTIVE") == 0) {
    schedule_policy = SchedulingPolicy::ADAPTIVE;

  } else if (FLAGS_htap_mode.compare("COLOC") == 0) {
    schedule_policy = SchedulingPolicy::S1_COLOCATED;

  } else if (FLAGS_htap_mode.compare("HYBRID-ISOLATED") == 0) {
    schedule_policy = SchedulingPolicy::S3_IS;

  } else if (FLAGS_htap_mode.compare("HYBRID-COLOC") == 0) {
    schedule_policy = SchedulingPolicy::S3_NI;
  }

  oltp_engine.snapshot();
  if (schedule_policy != SchedulingPolicy::S1_COLOCATED) {
    oltp_engine.etl(OLAP_socket);
  }

  HTAPSequenceConfig htap_conf(olap_nodes, oltp_nodes, FLAGS_adaptive_ratio,
                               FLAGS_oltp_elastic_threshold,
                               FLAGS_oltp_coloc_threshold, schedule_policy);

  if (FLAGS_htap_mode.compare("REMOTE-READ") == 0) {
    htap_conf.schedule_policy = SchedulingPolicy::CUSTOM;
    htap_conf.resource_policy = SchedulingPolicy::ISOLATED;
    htap_conf.data_access_policy = SchedulingPolicy::REMOTE_READ;
  }

  if (FLAGS_micro_ch_query > 0) {
    LOG(INFO) << "QQQQ: " << FLAGS_micro_ch_query;
    htap_conf.setChMicro(FLAGS_micro_ch_query);
  }

  std::vector<OLAPSequence> olap_clients;

  for (int i = 0; i < FLAGS_num_olap_clients; i++) {
    olap_clients.emplace_back(
        i, htap_conf, (FLAGS_gpu_olap ? DeviceType::GPU : DeviceType::CPU));
  }

  if (FLAGS_run_oltp) {
    oltp_engine.run();
    usleep(2000000);  // stabilize
    oltp_engine.print_differential_stats();
  }

  // usleep(2000000);  // stabilize

  exec_location{nodes[OLAP_socket]}.activate();

  profiling::resume();

  if (FLAGS_run_olap) {
    for (auto &olap_cl : olap_clients) {
      if (FLAGS_etl_interval_ms > 0) {
        olap_cl.execute(oltp_engine, FLAGS_num_olap_repeat,
                        FLAGS_per_query_snapshot, FLAGS_etl_interval_ms);
      } else {
        olap_cl.execute(oltp_engine, FLAGS_num_olap_repeat,
                        FLAGS_per_query_snapshot);
      }

      // LOG(INFO) << olap_cl;
      LOG(INFO) << "OLAP client#" << olap_cl.client_id << " finished.";
    }
    profiling::pause();

    // LOG(INFO) << olap_queries;
  }

  oltp_engine.print_global_stats();
  olap_clients.clear();

  if (!FLAGS_run_oltp) {
    // FIXME: hack because it needs to run before it can be stopped
    oltp_engine.run();
  }

  // profiling::pause();

  LOG(INFO) << "Shutdown initiated";

  // OLTP
  oltp_engine.shutdown();

  delete bench;

  return 0;

  // FLAGS_num_olap_repeat =
  //(NUM_OLAP_REPEAT / FLAGS_num_olap_clients);  // warmmup
}
