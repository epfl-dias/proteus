/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include "ndp/cluster_command_provider.hpp"

#include <unistd.h>

#include <cli-flags.hpp>
#include <command-provider/local-command-provider.hpp>
#include <common/common.hpp>
#include <distributed-runtime/cluster-manager.hpp>
#include <map>
#include <olap/plan/query-result.hpp>
#include <storage/mmap-file.hpp>

static auto catalogJSON = "inputs";

class unlink_upon_exit {
  size_t query;
  std::string label_prefix;

  std::string last_label;

  std::unique_ptr<QueryResult> last_result;

 public:
  unlink_upon_exit()
      : query(0),
        label_prefix("raw_server_" + std::to_string(getpid()) + "_q"),
        last_label("") {}

  unlink_upon_exit(size_t unique_id)
      : query(0),
        label_prefix("raw_server_" + std::to_string(unique_id) + "_q"),
        last_label("") {}

  [[nodiscard]] std::string get_label() const { return last_label; }

  std::string inc_label() {
    last_label = label_prefix + std::to_string(query++);
    return last_label;
  }

  void store(QueryResult&& qr) {
    last_result = std::make_unique<QueryResult>(std::move(qr));
  }

  void reset() { last_result.reset(); }
};

class ClusterCommandProvider::impl {
 public:
  unlink_upon_exit uue{};
  std::vector<std::string> prepared_statements{};
  // std::map<std::string, PreparedStatement> preparedStatements;
};

std::string ClusterCommandProvider::prepareStatement(const fs::path& plan) {
  std::string label = this->p_impl->uue.inc_label();

  LOG(INFO) << "PrepareStatement:: Label:  " << label;

  ctx.getClusterManager().broadcastPrepareQuery(
      {label, std::make_unique<mmap_file>(plan, PAGEABLE)});
  p_impl->prepared_statements.emplace_back(label);
  return label;
}

std::string ClusterCommandProvider::prepareStatement(
    const std::span<const std::byte>& plan) {
  std::string label = this->p_impl->uue.inc_label();

  LOG(INFO) << "PrepareStatement:: Label:  " << label;

  ctx.getClusterManager().broadcastPrepareQuery({label, plan});
  p_impl->prepared_statements.emplace_back(label);
  return label;
}
void ClusterCommandProvider::prepareStatement(
    const std::string& label, const std::span<const std::byte>& plan) {
  LOG(INFO) << "PrepareStatement:: Label:  " << label;
  ctx.getClusterManager().broadcastPrepareQuery({label, plan});
  p_impl->prepared_statements.emplace_back(label);
}

fs::path ClusterCommandProvider::runPreparedStatement(const std::string& label,
                                                      bool echo) {
  ctx.getClusterManager().broadcastExecuteQuery(label);
  return fs::path(getResultLocation(label));
}

fs::path ClusterCommandProvider::runStatement(const fs::path& plan, bool echo) {
  std::string label = this->p_impl->uue.inc_label();
  ctx.getClusterManager().broadcastPrepareExecuteQuery(
      {label, std::make_unique<mmap_file>(plan, PAGEABLE)});
  p_impl->prepared_statements.emplace_back(label);

  return fs::path(getResultLocation(label));
}

fs::path ClusterCommandProvider::runStatement(
    const std::span<const std::byte>& plan, bool echo) {
  std::string label = this->p_impl->uue.inc_label();
  ctx.getClusterManager().broadcastPrepareExecuteQuery({label, plan});
  p_impl->prepared_statements.emplace_back(label);

  return fs::path(getResultLocation(label));
}

std::string ClusterCommandProvider::getResultLocation(
    const std::string& label) {
  std::vector<proteus::distributed::QueryStatus> query_status;

  auto pr_id = ctx.getClusterManager().getResultServerId();
  while (true) {
    query_status = ctx.getClusterManager().getQueryStatus(label, pr_id);
    if (query_status.size() > 0) {
      break;
    }
  }
  std::string result_location;
  if (query_status[0].status == proteus::distributed::QueryStatus::EXECUTED) {
    result_location = query_status[0].result_location;
  } else {
    LOG(ERROR) << "Query status is not executed, but: "
               << query_status[0].status;
  }
  return result_location;
}

ClusterCommandProvider::ClusterCommandProvider(proteus::ndp& ctx)
    : p_impl(std::make_unique<ClusterCommandProvider::impl>()), ctx(ctx) {}
ClusterCommandProvider::~ClusterCommandProvider() = default;
