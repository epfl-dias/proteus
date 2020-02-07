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

#ifndef PROTEUS_OLAP_SEQUENCE_HPP
#define PROTEUS_OLAP_SEQUENCE_HPP

#include <deque>
#include <plan/prepared-statement.hpp>
#include <topology/device-types.hpp>
#include <topology/topology.hpp>
#include <vector>

#include "oltp.hpp"
#include "prepared-query.hpp"

using exec_nodes = std::vector<topology::numanode *>;
typedef std::chrono::time_point<std::chrono::system_clock> timepoint_t;

namespace SchedulingPolicy {

enum DataAccess { LOCAL_READ, REMOTE_READ, HYBRID_READ };

enum ResourceSchedule { ISOLATED, COLOCATED, ELASTIC };

enum ScheduleMode { S1_COLOCATED, S2_ISOLATED, S3_IS, S3_NI, ADAPTIVE, CUSTOM };

};  // namespace SchedulingPolicy

struct OLAPSequenceStats {
  const uint run_id;
  // runtimes[# queries][# runs] milliseconds
  std::vector<std::vector<std::pair<size_t, size_t>>> sts;
  std::vector<std::vector<size_t>> input_records;
  std::vector<std::vector<std::pair<double, double>>> freshness_ratios;
  std::vector<std::vector<std::pair<double, double>>> oltp_stats;
  std::vector<std::vector<SchedulingPolicy::ScheduleMode>> sts_state;

  std::vector<std::pair<size_t, size_t>> sequence_time;

  OLAPSequenceStats(uint run_id, size_t number_of_queries, size_t repeat)
      : run_id(run_id) {
    sts.reserve(number_of_queries);
    for (size_t i = 0; i < number_of_queries; i++) {
      sts.emplace_back();
      sts[i].reserve(repeat);

      sts_state.emplace_back();
      sts_state[i].reserve(repeat);

      oltp_stats.emplace_back();
      oltp_stats[i].reserve(repeat);

      freshness_ratios.emplace_back();
      freshness_ratios[i].reserve(repeat);

      input_records.emplace_back();
      input_records[i].reserve(repeat);
    }

    sequence_time.reserve(repeat);
  }
};

class HTAPSequenceConfig {
 public:
  const exec_nodes olap_nodes;
  const exec_nodes oltp_nodes;
  const uint oltp_scale_threshold;
  const uint collocated_worker_threshold;

  SchedulingPolicy::ScheduleMode schedule_policy;
  SchedulingPolicy::ResourceSchedule resource_policy;
  SchedulingPolicy::DataAccess data_access_policy;

  bool ch_micro;
  uint q_num;

  HTAPSequenceConfig(exec_nodes olap_nodes, exec_nodes oltp_nodes,
                     uint oltp_scale_threshold = 0,
                     uint collocated_worker_threshold = 0,
                     SchedulingPolicy::ScheduleMode schedule_policy =
                         SchedulingPolicy::S2_ISOLATED);

  HTAPSequenceConfig(exec_nodes olap_nodes, exec_nodes oltp_nodes,
                     uint oltp_scale_threshold = 0,
                     uint collocated_worker_threshold = 0,
                     SchedulingPolicy::ResourceSchedule resource_policy =
                         SchedulingPolicy::ISOLATED,
                     SchedulingPolicy::DataAccess data_access_policy =
                         SchedulingPolicy::LOCAL_READ);

  void setChMicro(uint q_num) {
    ch_micro = true;
    this->q_num = q_num;
  }
};

class OLAPSequence {
 private:
  int total_queries;
  std::vector<PreparedQuery> stmts;
  std::deque<OLAPSequenceStats *> stats;
  HTAPSequenceConfig conf;

 public:
  OLAPSequence(int client_id, HTAPSequenceConfig conf, DeviceType dev);
  void execute(OLTP &txn_engine, int repeat = 1, bool per_query_snapshot = true,
               size_t etl_interval_ms = std::numeric_limits<size_t>::max());

  friend std::ostream &operator<<(std::ostream &, const OLAPSequence &);
  friend std::ostream &operator<<(std::ostream &, const OLAPSequenceStats &);

  // TODO: implement
  //~OLAPSequence()

 private:
  void migrateState(SchedulingPolicy::ScheduleMode &curr,
                    SchedulingPolicy::ScheduleMode to, OLTP &txn_engine);
  SchedulingPolicy::ScheduleMode getNextState(
      SchedulingPolicy::ScheduleMode current_state,
      const std::pair<double, double> &freshness_ratios);
  std::pair<double, double> getFreshnessRatios(OLTP &txn_engine,
                                               size_t &query_idx,
                                               OLAPSequenceStats *stats_local);
  void setupAdaptiveSequence();
  auto getIsolatedOLAPResources();
  auto getColocatedResources();
  auto getElasticResources();

  static constexpr size_t n_warmup_queries = 1;

 public:
  const int client_id;
};

std::ostream &operator<<(std::ostream &, const OLAPSequence &);
std::ostream &operator<<(std::ostream &, const OLAPSequenceStats &);

#endif /* PROTEUS_OLAP_SEQUENCE_HPP */
