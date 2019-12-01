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

#ifndef HARMONIA_QUERY_SEQUENCE_HPP_
#define HARMONIA_QUERY_SEQUENCE_HPP_

#include <functional>
#include <string>
#include <vector>

//#include "adaptors/aeolus-plugin.hpp"
// #include "codegen/memory/block-manager.hpp"
// #include "codegen/memory/memory-manager.hpp"
#include "plan/prepared-statement.hpp"
// #include "codegen/storage/storage-manager.hpp"
#include "topology/affinity_manager.hpp"
// #include "util/jit/pipeline.hpp"
// #include "codegen/util/parallel-context.hpp"
// #include "codegen/util/profiling.hpp"
// #include "codegen/util/timing.hpp"
#include "query-interface.hpp"
#include "routing/affinitizers.hpp"
#include "routing/degree-of-parallelism.hpp"

namespace htap {
namespace queries {

class QuerySequence {
 public:
  const uint elastic_resources;
  const bool trade_resources;
  const bool per_query_freshness;
  const std::function<void()> txn_snapshot_agent;
  const std::function<void()> etl_agent;
  const size_t sequence_id;

 private:
  std::vector<PreparedStatement> queries;
  static size_t sequence_counter;

  // static const auto query_sequence{q_ch1};

 public:
  // std::vector<std::function<PreparedStatement(
  //        DegreeOfParallelism dop, std::unique_ptr<Affinitizer> aff_parallel,
  //        std::unique_ptr<Affinitizer> aff_reduce)>>
  //        query_builders,

  QuerySequence(const topology::cpunumanode &olap_node,
                const topology::cpunumanode &oltp_node,
                std::function<void()> txn_snapshot_agent,
                std::function<void()> etl_agent, uint elastic_resources = 0,
                bool trade_resources = false, bool per_query_freshness = false);
  void execute(bool log_results = false, bool warmup = false);

  static bool should_etl() { assert(false && "Not-implemented"); }
};
}  // namespace queries
}  // namespace htap

#endif /* HARMONIA_QUERY_SEQUENCE_HPP_ */
