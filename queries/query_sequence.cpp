/*
    Harmonia -- High-performance elastic HTAP on heterogeneous hardware.

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

#include "query_sequence.hpp"

#include "codegen/plan/prepared-statement.hpp"
#include "queries.hpp"
#include "routing/affinitizers.hpp"
#include "routing/degree-of-parallelism.hpp"

namespace harmonia {

size_t QuerySequence::sequence_counter = 0;

QuerySequence::QuerySequence(
    const topology::cpunumanode &olap_node,
    const topology::cpunumanode &oltp_node,
    std::vector<std::function<PreparedStatement(
        DegreeOfParallelism dop, std::unique_ptr<Affinitizer> aff_parallel,
        std::unique_ptr<Affinitizer> aff_reduce)>>
        query_builders,
    std::function<void()> txn_snapshot_agent, std::function<void()> etl_agent,
    uint elastic_resources, bool trade_resources, bool per_query_freshness)
    : sequence_id(sequence_counter++),
      elastic_resources(elastic_resources),
      trade_resources(trade_resources),
      per_query_freshness(per_query_freshness) {
  time_block t("T_harmonia_QuerySequence::init: ");

  LOG(INFO) << "Sequence " << this->sequence_id << ": prepare_open";

  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;

  uint j = 0;
  for (auto id : olap_node.local_cores) {
    if (trade_resources && elastic_resources > 0 && j < elastic_resources) {
      j++;
      continue;
    }
    coreids.emplace_back(id);
  }

  if (elastic_resources > 0) {
    uint i = 0;
    for (auto id : oltp_node.local_cores) {
      coreids.emplace_back(id);
      if (++i >= elastic_resources) {
        break;
      }
    }

    if (trade_resources) {
      for (auto id : olap_node.local_cores) {
        coreids.emplace_back(id);
      }
    }
  }

  // DegreeOfParallelism dop{coreids.size()};
  // for (const auto &q : query_builders) {  // q_ch6
  //   // std::unique_ptr<Affinitizer> aff_parallel =
  //   //     std::make_unique<CpuCoreAffinitizer>();

  //   std::unique_ptr<Affinitizer> aff_parallel =
  //       std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
  //   std::unique_ptr<Affinitizer> aff_reduce =
  //       std::make_unique<CpuCoreAffinitizer>();

  //   queries.emplace_back(
  //       q(dop, std::move(aff_parallel), std::move(aff_reduce)));
  // }

  DegreeOfParallelism dop{coreids.size()};

  auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
    return std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
  };

  auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
    return std::make_unique<CpuCoreAffinitizer>();
  };

  typedef decltype(aff_parallel) aff_t;
  typedef decltype(aff_reduce) red_t;

  // for (const auto &q : {q_ch1<aff_t, red_t, plugin_t>, q_ch6<aff_t, red_t,
  // plugin_t>,
  //                       q_ch19<aff_t, red_t, plugin_t>}) {
  using plugin_t = AeolusLocalPlugin;
  for (const auto &q : {q_ch19<aff_t, red_t, plugin_t>}) {
    // std::unique_ptr<Affinitizer> aff_parallel =
    //     std::make_unique<CpuCoreAffinitizer>();

    this->queries.emplace_back(q(dop, aff_parallel, aff_reduce));
  }

  LOG(INFO) << "Sequence " << this->sequence_id << ": prepare_close";
}

void QuerySequence::execute(bool log_results, bool warmup) {
  if (warmup) {
    LOG(INFO) << "Sequence " << this->sequence_id << ": warmup_open";
    queries[0].execute();
    LOG(INFO) << "Sequence " << this->sequence_id << ": warmup_close";
  }

  {
    time_block t("T_harmonia_QuerySequence::execute: ");
    LOG(INFO) << "Sequence " << this->sequence_id << ": open";
    if (log_results) {
      for (auto &q : queries) {
        LOG(INFO) << q.execute();
      }
    } else {
      for (auto &q : queries) {
        q.execute();
      }
    }
    LOG(INFO) << "Sequence " << this->sequence_id << ": close";
  }
}
}  // namespace harmonia
