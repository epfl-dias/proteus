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

#include "olap-sequence.hpp"

#include <aeolus-plugin.hpp>
#include <numeric>
#include <routing/affinitizers.hpp>
#include <routing/degree-of-parallelism.hpp>
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>
#include <type_traits>
#include <util/timing.hpp>

#include "../htap-cli-flags.hpp"
#include "ch/ch-queries.hpp"

// FIXME: remove following
#define ETL_AFFINITY_SOCKET 0

const auto q01_rel = std::vector<std::string>{"tpcc_orderline"};
const auto q02_rel = std::vector<std::string>{
    "tpcc_region", "tpcc_nation", "tpcc_supplier", "tpcc_stock", "tpcc_item"};
const auto q04_rel = std::vector<std::string>{"tpcc_order", "tpcc_orderline"};
const auto q06_rel = std::vector<std::string>{"tpcc_orderline"};
const auto q19_rel = std::vector<std::string>{"tpcc_item", "tpcc_orderline"};

typedef std::function<std::unique_ptr<Affinitizer>()> red_t;

typedef std::function<PreparedStatement(DegreeOfParallelism, aff_t, red_t)>
    prep_t;

typedef std::function<prep_t(pg)> prep_wrapper_t;

prep_wrapper_t qs(const std::string &plan) {
  return [plan](pg type) {
    return [plan, type](DegreeOfParallelism dop, aff_t aff_parallel,
                        red_t aff_reduce, DeviceType dev = DeviceType::CPU) {
      const std::string planDir = "benchmarks/htap/queries/clotho/cpu/";
      return PreparedStatement::from(planDir + plan, plan);
    };
  };
}

template <size_t id, typename plugin_t>
prep_t qs_old_inner() {
  return [](DegreeOfParallelism dop, aff_t aff_parallel, red_t aff_reduce,
            DeviceType dev = DeviceType::CPU) {
    return Q<id>::template prepare<plugin_t, aff_t, red_t>(dop, aff_parallel,
                                                           aff_reduce);
  };
}

template <size_t id>
prep_wrapper_t qs_old() {
  return [](pg type) {
    auto t = type.getType();
    if (t == AeolusLocalPlugin::type) {
      return qs_old_inner<id, AeolusLocalPlugin>();
    } else if (t == AeolusRemotePlugin::type) {
      return qs_old_inner<id, AeolusRemotePlugin>();
    } else if (t == AeolusElasticPlugin::type) {
      return qs_old_inner<id, AeolusElasticPlugin>();
    } else if (t == AeolusElasticNIPlugin::type) {
      return qs_old_inner<id, AeolusElasticNIPlugin>();
    } else if (t == AeolusCowPlugin::type) {
      return qs_old_inner<id, AeolusCowPlugin>();
    } else {
      assert(false);
      throw std::runtime_error("Unknown plugin: " + t);
    }
  };
}

std::vector<std::pair<std::vector<std::string>, prep_wrapper_t>> ch_map = {
    //    {q01_rel, qs("Q01.sql.json")},
    //    {q02_rel, qs("Q02_simplified.sql.json")},
    //    {q02_rel, qs("Q02_simplified_red.sql.json")},
    ////    {q03_rel, qs("Q03_simplified.sql.json")},
    //    {q04_rel, qs("Q04.sql.json")},
    //    {q06_rel, qs("Q06.sql.json")},
    //    {q09_rel, qs("Q09_simplified.sql.json")},
    //    {q12_rel, qs("Q12.sql.json")},
    //    {q15_rel, qs("Q15.sql.json")},
    //    {q18_rel, qs("Q18.sql.json")},
    //    {q19_rel, qs("Q19_simplified.sql.json")},
    {q01_rel, qs_old<1>()},
    //    {q04_rel, qs_old<4, plugin_t>()},
    {q06_rel, qs_old<6>()},
    {q19_rel, qs_old<19>()},
};

auto OLAPSequence::getIsolatedOLAPResources() {
  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;
  for (auto &olap_n : conf.olap_nodes) {
    for (auto id :
         (dynamic_cast<topology::cpunumanode *>(olap_n))->local_cores) {
      coreids.emplace_back(id);
    }
  }
  return coreids;
}

auto OLAPSequence::getColocatedResources() {
  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;
  bool skip = true;
  for (auto &olap_n : conf.olap_nodes) {
    for (auto id :
         (dynamic_cast<topology::cpunumanode *>(olap_n))->local_cores) {
      skip = !skip;
      if (skip) continue;
      coreids.emplace_back(id);
    }
  }

  skip = true;
  for (auto &oltp_n : conf.oltp_nodes) {
    for (auto id :
         (dynamic_cast<topology::cpunumanode *>(oltp_n))->local_cores) {
      skip = !skip;
      if (skip) continue;

      coreids.emplace_back(id);
    }
  }
  return coreids;
}

auto OLAPSequence::getElasticResources() {
  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;

  for (auto &olap_n : conf.olap_nodes) {
    for (auto id :
         (dynamic_cast<topology::cpunumanode *>(olap_n))->local_cores) {
      coreids.emplace_back(id);
    }
  }

  int k = 0;
  for (auto &oltp_n : conf.oltp_nodes) {
    for (auto id :
         (dynamic_cast<topology::cpunumanode *>(oltp_n))->local_cores) {
      if (k >= conf.oltp_scale_threshold) {
        break;
      }
      coreids.emplace_back(id);
      k++;
    }

    if (k >= conf.oltp_scale_threshold) {
      break;
    }
  }
  assert(k == conf.oltp_scale_threshold);

  return coreids;
}

void OLAPSequence::setupAdaptiveSequence() {
  auto resources_colocated = getColocatedResources();
  auto resources_isolated = getIsolatedOLAPResources();
  auto resources_elastic = getElasticResources();

  DegreeOfParallelism colocated_dop{resources_colocated.size()};
  DegreeOfParallelism isolated_dop{resources_isolated.size()};
  DegreeOfParallelism elastic_dop{resources_elastic.size()};

  // ---- Queries -----
  // S1_COLOCATED

  {
    auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<SpecificCpuCoreAffinitizer>(resources_colocated);
    };

    auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<CpuCoreAffinitizer>();
    };

    for (const auto &q : ch_map) {
      stmts.emplace_back(q.second(pg(AeolusRemotePlugin::type))(
          colocated_dop, aff_parallel, aff_reduce));
      total_queries++;
    }
  }

  // S2_ISOLATED

  {
    auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<SpecificCpuCoreAffinitizer>(resources_isolated);
    };

    auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<CpuCoreAffinitizer>();
    };

    for (const auto &q : ch_map) {
      stmts.emplace_back(q.second(pg(AeolusLocalPlugin::type))(
          isolated_dop, aff_parallel, aff_reduce));
    }
  }

  // S3_IS

  {
    auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<SpecificCpuCoreAffinitizer>(resources_isolated);
    };

    auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<CpuCoreAffinitizer>();
    };

    for (const auto &q : ch_map) {
      stmts.emplace_back(q.second(pg(AeolusElasticPlugin::type))(
          isolated_dop, aff_parallel, aff_reduce));
    }
  }

  // S3_NI

  {
    auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<SpecificCpuCoreAffinitizer>(resources_elastic);
    };

    auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
      return std::make_unique<CpuCoreAffinitizer>();
    };

    for (const auto &q : ch_map) {
      stmts.emplace_back(q.second(pg(AeolusElasticNIPlugin::type))(
          elastic_dop, aff_parallel, aff_reduce));
    }
  }

  return;
}

OLAPSequence::OLAPSequence(int client_id, HTAPSequenceConfig conf,
                           DeviceType dev)
    : conf(conf), total_queries(0), client_id(client_id) {
  if (dev == DeviceType::GPU &&
      conf.schedule_policy != SchedulingPolicy::S1_COLOCATED) {
    assert(false && "GPU OLAP supports isolated scheduling mode only");
  }

  if (conf.schedule_policy == SchedulingPolicy::ADAPTIVE) {
    setupAdaptiveSequence();
    return;
  }

  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;

  // Resource Scheduling
  if (dev == DeviceType::CPU) {
    if (conf.schedule_policy == SchedulingPolicy::S2_ISOLATED ||
        conf.schedule_policy == SchedulingPolicy::S3_IS ||
        conf.resource_policy == SchedulingPolicy::ISOLATED) {
      coreids = getIsolatedOLAPResources();
    } else if (conf.schedule_policy == SchedulingPolicy::S1_COLOCATED ||
               conf.resource_policy == SchedulingPolicy::COLOCATED) {
      coreids = getColocatedResources();

    } else if (conf.schedule_policy == SchedulingPolicy::S3_NI ||
               conf.resource_policy == SchedulingPolicy::ELASTIC) {
      assert(conf.oltp_scale_threshold > 0);
      coreids = getElasticResources();
    }
  }

  LOG(INFO) << "OLAP Cores:";
  for (const auto &cr : coreids) {
    LOG(INFO) << cr;
  }

  DegreeOfParallelism dop{(dev == DeviceType::CPU) ? coreids.size()
                                                   : conf.olap_nodes.size()};

  auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
    if (dev == DeviceType::CPU) {
      return std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
    } else {
      return std::make_unique<GPUAffinitizer>();
    }
  };

  auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
    return std::make_unique<CpuCoreAffinitizer>();
  };
  // Data Access Methods

  if (conf.data_access_policy == SchedulingPolicy::REMOTE_READ ||
      conf.schedule_policy == SchedulingPolicy::S1_COLOCATED) {
    for (const auto &q : ch_map) {
      stmts.emplace_back(q.second(pg(AeolusRemotePlugin::type))(
          dop, aff_parallel, aff_reduce));
      total_queries++;
    }

  } else if (conf.data_access_policy == SchedulingPolicy::LOCAL_READ ||
             conf.schedule_policy == SchedulingPolicy::S2_ISOLATED) {
    for (const auto &q : ch_map) {
      stmts.emplace_back(
          q.second(pg(AeolusLocalPlugin::type))(dop, aff_parallel, aff_reduce));
      total_queries++;
    }

  } else if (conf.data_access_policy == SchedulingPolicy::HYBRID_READ ||
             conf.schedule_policy == SchedulingPolicy::S3_IS ||
             conf.schedule_policy == SchedulingPolicy::S3_NI) {
    for (const auto &q : ch_map) {
      stmts.emplace_back(q.second(pg(AeolusElasticPlugin::type))(
          dop, aff_parallel, aff_reduce));
      total_queries++;
    }

  } else {
    assert(false && "Unsupported data access plugin");
  }
}

std::ostream &operator<<(std::ostream &out, const OLAPSequence &seq) {
  out << "OLAP Stats" << std::endl;
  for (const auto &r : seq.stats) {
    out << "Sequence Run # " << r->run_id << std::endl;

    // per query avg time
    // std::vector<double> q_avg_time(r->sts.size(), 0.0);
    uint i = 0;
    for (const auto &q : r->sts) {
      out << "\t\tQuery # " << (i + 1) << " -";
      for (const auto &q_ts : q) {
        // q_avg_time[i] += q_ts;

        out << "\t" << q_ts.second;
      }
      out << std::endl;
      // q_avg_time[i] /= q.size();
      // out << "\t\tQuery (Avg) # " << (i + 1) << " - " << q_avg_time[i] << "
      // ms"
      //     << std::endl;
      i++;
    }

    // average sequence time
    // double average = (double)(std::accumulate(r->sequence_time.begin(),
    //                                           r->sequence_time.end(), 0.0))
    //                                           /
    //                  ((double)r->sequence_time.size());

    // out << "\t\tQuery Sequence - " << average << " ms" << std::endl;

    out << "\tSequence Time "
        << " -";
    for (const auto &st : r->sequence_time) {
      out << "\t" << st.second;
    }
    out << std::endl;
  }
  return out;
}

std::ostream &operator<<(std::ostream &out, const OLAPSequenceStats &r) {
  out << "Sequence Run # " << r.run_id << std::endl;

  uint ts = 0;
  for (const auto &q : r.sts) {
    out << "\t\t\tTime -";
    for (const auto &q_ts : q) {
      out << "\t" << q_ts.first;
    }
    out << std::endl;
    ts++;
  }

  uint t = 0;
  for (const auto &q : r.oltp_stats) {
    out << "\t\t\tOLTP -";
    for (const auto &q_ts : q) {
      out << "\t" << q_ts.second;
    }
    out << std::endl;
    t++;
  }

  uint i = 0;
  for (const auto &q : r.sts) {
    out << "\t\tQuery # " << (i + 1) << " -";
    for (const auto &q_ts : q) {
      out << "\t" << q_ts.second;
    }
    out << std::endl;
    i++;
  }

  uint l = 0;
  for (const auto &q : r.freshness_ratios) {
    out << "\t\tR_fq " << (l + 1) << " -";
    for (const auto &q_ts : q) {
      out << "\t" << (q_ts.first);
    }
    out << std::endl;
    l++;
  }

  l = 0;
  for (const auto &q : r.freshness_ratios) {
    out << "\t\tR_ft " << (l + 1) << " -";
    for (const auto &q_ts : q) {
      out << "\t" << (q_ts.second);
    }
    out << std::endl;
    l++;
  }

  uint k = 0;
  for (const auto &q : r.sts_state) {
    out << "\t\tQState # " << (k + 1) << " -";
    for (const auto &q_ts : q) {
      out << "\t" << q_ts;
    }
    out << std::endl;
    k++;
  }

  out << "\tSequence Time "
      << " -";
  for (const auto &st : r.sequence_time) {
    out << "\t" << st.first;
  }
  out << std::endl;

  out << "\tSequence Query Time "
      << " -";
  for (const auto &st : r.sequence_time) {
    out << "\t" << st.second;
  }
  out << std::endl;

  return out;
}

HTAPSequenceConfig::HTAPSequenceConfig(
    exec_nodes olap_nodes, exec_nodes oltp_nodes, double adaptivity_ratio,
    uint oltp_scale_threshold, uint collocated_worker_threshold,
    SchedulingPolicy::ScheduleMode schedule_policy)
    : olap_nodes(olap_nodes),
      oltp_nodes(oltp_nodes),
      adaptivity_ratio(adaptivity_ratio),
      oltp_scale_threshold(oltp_scale_threshold),
      collocated_worker_threshold(collocated_worker_threshold),
      schedule_policy(schedule_policy) {
  switch (schedule_policy) {
    case SchedulingPolicy::S1_COLOCATED:
      resource_policy = SchedulingPolicy::COLOCATED;
      data_access_policy = SchedulingPolicy::REMOTE_READ;
      break;
    case SchedulingPolicy::S2_ISOLATED:
      resource_policy = SchedulingPolicy::ISOLATED;
      data_access_policy = SchedulingPolicy::LOCAL_READ;
      break;
    case SchedulingPolicy::S3_IS:
      resource_policy = SchedulingPolicy::ISOLATED;
      data_access_policy = SchedulingPolicy::HYBRID_READ;
      break;
    case SchedulingPolicy::S3_NI:
      resource_policy = SchedulingPolicy::ELASTIC;
      data_access_policy = SchedulingPolicy::HYBRID_READ;
      break;
    case SchedulingPolicy::ADAPTIVE:
    default:
      break;
  }
}
HTAPSequenceConfig::HTAPSequenceConfig(
    exec_nodes olap_nodes, exec_nodes oltp_nodes, double adaptivity_ratio,
    uint oltp_scale_threshold, uint collocated_worker_threshold,
    SchedulingPolicy::ResourceSchedule resource_policy,
    SchedulingPolicy::DataAccess data_access_policy)
    : olap_nodes(olap_nodes),
      oltp_nodes(oltp_nodes),
      adaptivity_ratio(adaptivity_ratio),
      oltp_scale_threshold(oltp_scale_threshold),
      collocated_worker_threshold(collocated_worker_threshold),
      resource_policy(resource_policy),
      data_access_policy(data_access_policy) {
  // set the schedule policy accordingly.
  switch (resource_policy) {
    case SchedulingPolicy::COLOCATED: {
      switch (data_access_policy) {
        case SchedulingPolicy::REMOTE_READ:
          schedule_policy = SchedulingPolicy::S1_COLOCATED;
          break;
        case SchedulingPolicy::HYBRID_READ:
        case SchedulingPolicy::LOCAL_READ:
        default:
          schedule_policy = SchedulingPolicy::CUSTOM;
          break;
      }
      break;
    }
    case SchedulingPolicy::ELASTIC: {
      assert(oltp_scale_threshold != 0 &&
             "Cannot have elastic OLAP with non-elastic OLTP!");
      switch (data_access_policy) {
        case SchedulingPolicy::HYBRID_READ:
          schedule_policy = SchedulingPolicy::S3_NI;
          break;
        case SchedulingPolicy::REMOTE_READ:
        case SchedulingPolicy::LOCAL_READ:
        default:
          schedule_policy = SchedulingPolicy::CUSTOM;
          break;
      }
      break;
    }
    case SchedulingPolicy::ISOLATED: {
      switch (data_access_policy) {
        case SchedulingPolicy::LOCAL_READ:
          schedule_policy = SchedulingPolicy::S2_ISOLATED;
          break;
        case SchedulingPolicy::HYBRID_READ:
          schedule_policy = SchedulingPolicy::S3_IS;
          break;
        case SchedulingPolicy::REMOTE_READ:
        default:
          schedule_policy = SchedulingPolicy::CUSTOM;
          break;
      }
      break;
    }
    default:
      schedule_policy = SchedulingPolicy::CUSTOM;
      break;
  }
}

void OLAPSequence::migrateState(SchedulingPolicy::ScheduleMode &curr,
                                SchedulingPolicy::ScheduleMode to,
                                OLTP &txn_engine) {
  // S1_COLOCATED, S2_ISOLATED, S3_IS, S3_NI
  if (curr == to) return;

  LOG(INFO) << "Migrating from state-" << curr << " to state-" << to;
  if ((curr == SchedulingPolicy::S2_ISOLATED &&
       to == SchedulingPolicy::S3_IS) ||
      (to == SchedulingPolicy::S2_ISOLATED &&
       curr == SchedulingPolicy::S3_IS)) {
    curr = to;
    return;
  }

  time_block t("T_HTAPstateMigration: ");

  // from isolated
  if ((curr == SchedulingPolicy::S2_ISOLATED ||
       curr == SchedulingPolicy::S3_IS) &&
      (to == SchedulingPolicy::S3_NI || to == SchedulingPolicy::S1_COLOCATED)) {
    if (to == SchedulingPolicy::S1_COLOCATED) {
      // migrate oltp worker
      assert(false && "colocated also requires data migration??");
      txn_engine.migrate_worker(conf.collocated_worker_threshold);

    } else {
      // scale down oltp
      txn_engine.scale_down(conf.oltp_scale_threshold);
    }
  }
  // to isolated
  else if ((curr == SchedulingPolicy::S3_NI ||
            curr == SchedulingPolicy::S1_COLOCATED) &&
           (to == SchedulingPolicy::S2_ISOLATED ||
            to == SchedulingPolicy::S3_IS)) {
    if (curr == SchedulingPolicy::S1_COLOCATED) {
      // migrate back
      assert(false && "colocated also requires data migration??");
      txn_engine.migrate_worker(conf.collocated_worker_threshold);
    } else {
      // scale-up oltp
      txn_engine.scale_up(conf.oltp_scale_threshold);
    }
  } else {
    // missing NI <-> COLOCATED
    LOG(INFO) << "Current State: " << curr;
    LOG(INFO) << "Next State: " << to;
    assert(false && "How come?");
  }

  curr = to;
}

std::pair<double, double> OLAPSequence::getFreshnessRatios(
    OLTP &txn_engine, size_t &query_idx, OLAPSequenceStats *stats_local) {
  // txn_engine.getFreshnessRatio();
  // getFreshnessRatioRelation(std::string table_name)

  // Need to capture information about the query,
  // somehow pass the metadata of query in this function, so that we know
  // exactly what this query will touch and what are the exact requirements.

  /*
  The ratio of fresh data to the overall amount of data that the query will
  access is given by Rf q , whereas the ratio of fresh data that the query will
  access to the overall number of fresh data is given by Rf t .
  */

  // BUG: account for number of columns in query and the datatype.
  // For the fresh tuples, account for the updates else inserts will be same no
  // matter what.

  auto db_f = txn_engine.getFreshness();
  auto query_f = txn_engine.getFreshnessRelation(ch_map[query_idx].first);

  auto query_fresh_data = query_f.second - query_f.first;
  auto total_fresh_data = db_f.second - db_f.first;

  // R_fq
  double r_fq = ((double)query_fresh_data) / ((double)query_f.second);
  double r_ft = ((double)query_fresh_data) / ((double)total_fresh_data);

  LOG(INFO) << "===========================";
  LOG(INFO) << "Query" << query_idx;
  LOG(INFO) << "Q_OLAP: " << query_f.first;
  LOG(INFO) << "Q_OLTP: " << query_f.second;
  LOG(INFO) << "D_OLAP: " << db_f.first;
  LOG(INFO) << "D_OLTP: " << db_f.second;

  LOG(INFO) << "query_fresh_data: " << query_fresh_data;
  LOG(INFO) << "total_fresh_data: " << total_fresh_data;
  LOG(INFO) << "R_FQ: " << r_fq;
  LOG(INFO) << "R_FT: " << r_ft;
  LOG(INFO) << "===========================";

  stats_local->input_records[query_idx].push_back(query_f.second);

  return std::make_pair(r_fq, r_ft);
}

SchedulingPolicy::ScheduleMode OLAPSequence::getNextState(
    SchedulingPolicy::ScheduleMode current_state,
    const std::pair<double, double> &freshness_ratios) {
  double r_fq = freshness_ratios.first;
  double r_ft = freshness_ratios.second;
  if (r_fq < (this->conf.adaptivity_ratio * r_ft)) {
    if (conf.oltp_scale_threshold <= 0) {
      return SchedulingPolicy::S3_IS;
    } else {
      return SchedulingPolicy::S3_NI;
    }

  } else {
    return SchedulingPolicy::S2_ISOLATED;
  }
}

static inline size_t time_diff(timepoint_t end, timepoint_t start) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
      .count();
}

void OLAPSequence::execute(OLTP &txn_engine, int repeat,
                           bool per_query_snapshot, size_t etl_interval_ms) {
  // assert(conf.schedule_policy != SchedulingPolicy::CUSTOM &&
  //        "Not supported currently");

  SchedulingPolicy::ScheduleMode current_state = SchedulingPolicy::S2_ISOLATED;
  if (conf.schedule_policy == SchedulingPolicy::S1_COLOCATED) {
    current_state = SchedulingPolicy::S1_COLOCATED;
  }

  static uint run = 0;
  OLAPSequenceStats *stats_local =
      new OLAPSequenceStats(++run, total_queries, repeat);

  timepoint_t last_etl = std::chrono::system_clock::now() -
                         std::chrono::milliseconds(etl_interval_ms);

  // txn_engine.snapshot();
  // if (current_state != SchedulingPolicy::S1_COLOCATED) {
  //   txn_engine.etl(ETL_AFFINITY_SOCKET);
  //   last_etl = std::chrono::system_clock::now();
  // }

  // warm-up
  LOG(INFO) << "Warm-up execution - BEGIN";
  for (uint i = 0; i < n_warmup_queries; i++) {
    stmts[i % stmts.size()].execute();
  }
  LOG(INFO) << "Warm-up execution - END";

  // sequence
  LOG(INFO) << "Exeucting OLAP Sequence[Client # " << this->client_id << "]";

  if (conf.schedule_policy != SchedulingPolicy::ADAPTIVE &&
      conf.schedule_policy != SchedulingPolicy::CUSTOM)
    migrateState(current_state, conf.schedule_policy, txn_engine);

  if (conf.schedule_policy == SchedulingPolicy::CUSTOM) {
    // FIXME: move to appropriate custom state.
    current_state = SchedulingPolicy::CUSTOM;
  }

  timepoint_t global_start = std::chrono::system_clock::now();

  for (int i = 0; i < repeat; i++) {
    LOG(INFO) << "Sequence # " << i;
    timepoint_t start_seq = std::chrono::system_clock::now();
    // add time based ETL here.

    if (!per_query_snapshot) {
      txn_engine.snapshot();
      if (current_state == SchedulingPolicy::S2_ISOLATED ||
          (conf.schedule_policy != SchedulingPolicy::S1_COLOCATED &&
           time_diff(std::chrono::system_clock::now(), last_etl) >=
               etl_interval_ms)) {
        txn_engine.etl(ETL_AFFINITY_SOCKET);
        last_etl = std::chrono::system_clock::now();
      }
    }
    for (size_t j = 0; j < total_queries; j++) {
      // update the snapshot
      if (per_query_snapshot) {
        txn_engine.snapshot();

        if (conf.schedule_policy != SchedulingPolicy::S2_ISOLATED ||
            conf.schedule_policy != SchedulingPolicy::S1_COLOCATED) {
          if (time_diff(std::chrono::system_clock::now(), last_etl) >=
              etl_interval_ms) {
            LOG(INFO) << "ETL_TIME:";
            txn_engine.etl(ETL_AFFINITY_SOCKET);
            last_etl = std::chrono::system_clock::now();
          }
        }
      }

      auto f_ratio = getFreshnessRatios(txn_engine, j, stats_local);
      if (conf.schedule_policy == SchedulingPolicy::ADAPTIVE)
        migrateState(current_state, getNextState(current_state, f_ratio),
                     txn_engine);

      if (current_state == SchedulingPolicy::S2_ISOLATED) {
        // have to do it anyway
        LOG(INFO) << "ETL_STATE:";
        txn_engine.etl(ETL_AFFINITY_SOCKET);
        last_etl = std::chrono::system_clock::now();
      }

      // Query Execution

      timepoint_t start = std::chrono::system_clock::now();

      if (conf.schedule_policy == SchedulingPolicy::ADAPTIVE) {
        // execute apprpriate query
        assert((j + (total_queries * current_state)) < stmts.size());
        LOG(INFO) << stmts[j + (total_queries * current_state)].execute();
      } else {
        LOG(INFO) << stmts[j].execute();
      }
      timepoint_t end = std::chrono::system_clock::now();

      stats_local->freshness_ratios[j].push_back(f_ratio);

      stats_local->oltp_stats[j].push_back(
          txn_engine.get_differential_stats(true));

      stats_local->sts[j].push_back(
          std::make_pair(time_diff(end, global_start), time_diff(end, start)));
      stats_local->sts_state[j].push_back(current_state);
    }

    // End sequence
    timepoint_t end_seq = std::chrono::system_clock::now();
    stats_local->sequence_time.push_back(std::make_pair(
        time_diff(end_seq, global_start), time_diff(end_seq, start_seq)));
  }

  LOG(INFO) << "Stats-Local:";
  std::cout << *stats_local << std::endl;
  LOG(INFO) << "Stats-Local-END";
  stats.push_back(stats_local);

  LOG(INFO) << "Exeucting  Sequence - END";
  txn_engine.print_storage_stats();
}

// void OLAPSequence::execute(OLTP &txn_engine, int repeat,
//                            bool per_query_snapshot, size_t etl_interval_ms) {
//   assert(conf.schedule_policy != SchedulingPolicy::CUSTOM &&
//          "Not supported currently");

//   SchedulingPolicy::ScheduleMode current_state =
//   SchedulingPolicy::S2_ISOLATED; static uint run = 0; OLAPSequenceStats
//   *stats_local =
//       new OLAPSequenceStats(++run, total_queries, repeat);

//   timepoint_t last_etl = std::chrono::system_clock::now() -
//                          std::chrono::milliseconds(etl_interval_ms);

//   // warm-up
//   LOG(INFO) << "Warm-up execution - BEGIN";
//   for (uint i = 0; i < n_warmup_queries; i++) {
//     stmts[i % stmts.size()].execute();
//   }
//   LOG(INFO) << "Warm-up execution - END";

//   // sequence
//   LOG(INFO) << "Exeucting OLAP Sequence[Client # " << this->client_id << "]";

//   if (conf.schedule_policy != SchedulingPolicy::ADAPTIVE)
//     migrateState(current_state, conf.schedule_policy, txn_engine);

//   timepoint_t global_start = std::chrono::system_clock::now();

//   for (int i = 0; i < repeat; i++) {
//     LOG(INFO) << "Sequence # " << i;
//     timepoint_t start_seq = std::chrono::system_clock::now();
//     // add time based ETL here.

//     for (size_t j = 0; j < total_queries; j++) {
//       // Query Execution

//       timepoint_t start = std::chrono::system_clock::now();

//       LOG(INFO) << stmts[j].execute(false);

//       timepoint_t end = std::chrono::system_clock::now();

//       stats_local->oltp_stats[j].push_back(
//           txn_engine.get_differential_stats(true));

//       stats_local->sts[j].push_back(
//           std::make_pair(time_diff(end, global_start), time_diff(end,
//           start)));
//       stats_local->sts_state[j].push_back(current_state);
//     }

//     // End sequence
//     timepoint_t end_seq = std::chrono::system_clock::now();
//     stats_local->sequence_time.push_back(std::make_pair(
//         time_diff(end_seq, global_start), time_diff(end_seq, start_seq)));
//   }

//   LOG(INFO) << "Stats-Local:";
//   LOG(INFO) << *stats_local;
//   LOG(INFO) << "Stats-Local-END";
//   stats.push_back(stats_local);

//   LOG(INFO) << "Exeucting  Sequence - END";
//   txn_engine.print_storage_stats();
// }
