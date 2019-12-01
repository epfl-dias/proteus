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

#include "queries/query-sequence.hpp"

#include "plan/prepared-statement.hpp"
#include "queries/query-interface.hpp"
#include "routing/affinitizers.hpp"
#include "routing/degree-of-parallelism.hpp"
#include "util/timing.hpp"

namespace htap {
namespace queries {

size_t QuerySequence::sequence_counter = 0;

QuerySequence::QuerySequence(const topology::cpunumanode &olap_node,
                             const topology::cpunumanode &oltp_node,
                             std::function<void()> txn_snapshot_agent,
                             std::function<void()> etl_agent,
                             uint elastic_resources, bool trade_resources,
                             bool per_query_freshness)
    : elastic_resources(elastic_resources),
      trade_resources(trade_resources),
      per_query_freshness(per_query_freshness),
      sequence_id(sequence_counter++) {
  time_block t("T_harmonia_QuerySequence::init: ");

  LOG(INFO) << "Sequence " << this->sequence_id << ": prepare_open";

  std::vector<SpecificCpuCoreAffinitizer::coreid_t> coreids;

  // TODO: Implement query sequence.

  // if (per_query_freshness) {
  //   uint j = 0;
  //   for (auto id : olap_node.local_cores) {
  //     if (trade_resources && elastic_resources > 0 && j < elastic_resources)
  //     {
  //       j++;
  //       continue;
  //     }
  //     coreids.emplace_back(id);
  //   }

  //   if (elastic_resources > 0) {
  //     uint i = 0;
  //     for (auto id : oltp_node.local_cores) {
  //       coreids.emplace_back(id);
  //       if (++i >= elastic_resources) {
  //         break;
  //       }
  //     }

  //     if (trade_resources) {
  //       for (auto id : olap_node.local_cores) {
  //         coreids.emplace_back(id);
  //       }
  //     }
  //   }

  //   DegreeOfParallelism dop{coreids.size()};

  //   auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
  //     return std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
  //   };

  //   auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
  //     return std::make_unique<CpuCoreAffinitizer>();
  //   };

  //   typedef decltype(aff_parallel) aff_t;
  //   typedef decltype(aff_reduce) red_t;

  //   for (const auto &q : {q_ch1<aff_t, red_t, AeolusElasticPlugin>,
  //                         q_ch6<aff_t, red_t, AeolusElasticPlugin>,
  //                         q_ch19<aff_t, red_t, AeolusElasticPlugin>}) {
  //     this->queries.emplace_back(q(dop, aff_parallel, aff_reduce));
  //   }

  // } else {
  //   // batch of queries with batch freshness.
  //   for (auto id : olap_node.local_cores) {
  //     coreids.emplace_back(id);
  //   }

  //   DegreeOfParallelism dop{coreids.size()};

  //   auto aff_parallel = [&]() -> std::unique_ptr<Affinitizer> {
  //     return std::make_unique<SpecificCpuCoreAffinitizer>(coreids);
  //   };

  //   auto aff_reduce = []() -> std::unique_ptr<Affinitizer> {
  //     return std::make_unique<CpuCoreAffinitizer>();
  //   };

  //   typedef decltype(aff_parallel) aff_t;
  //   typedef decltype(aff_reduce) red_t;

  //   for (const auto &q : {}) {
  //     this->queries.emplace_back(q(dop, aff_parallel, aff_reduce));
  //   }
  // }
  // }
  // else {
  //   assert(false && "why control reaches here?");
  // }

  LOG(INFO) << "Sequence " << this->sequence_id << ": prepare_close";
}

void QuerySequence::execute(bool log_results, bool warmup) {
  if (warmup) {
    LOG(INFO) << "Sequence " << this->sequence_id << ": warmup_open";
    queries[0].execute();
    LOG(INFO) << "Sequence " << this->sequence_id << ": warmup_close";
  }
  //
  {
    time_block t("T_harmonia_QuerySequence::execute: ");
    LOG(INFO) << "Sequence " << this->sequence_id << ": open";

    // TODO: add logic that if current olap snapshot is too stale, do an ETL
    // anyway. it will not effect the etl-local wones and it has to be done
    // but will largely effect the queries with hybrid executions.

    if (!per_query_freshness) {
      time_block t("T_harmonia_QuerySequence::ETL: ");
      this->etl_agent();
    }

    for (auto &q : queries) {
      if (per_query_freshness) {
        this->txn_snapshot_agent();
      }
      LOG(INFO) << q.execute();
    }

    LOG(INFO) << "Sequence " << this->sequence_id << ": close";
  }
}
}  // namespace queries
}  // namespace htap
