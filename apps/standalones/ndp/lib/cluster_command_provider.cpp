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
#include <distributed-runtime/cluster-manager.hpp>
#include <map>
#include <ndp/cluster_command_provider.hpp>
#include <olap/plan/prepared-statement.hpp>
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

struct ClusterCommandProvider::impl {
  unlink_upon_exit uue;
  std::vector<std::string> prepared_statements{};
  // std::map<std::string, PreparedStatement> preparedStatements;
};

std::string ClusterCommandProvider::prepareStatement(const fs::path& plan) {
  std::string label = p_impl->uue.inc_label();
  proteus::distributed::ClusterManager::getInstance().broadcastQuery(
      {label, std::make_unique<mmap_file>(plan, PAGEABLE)});

  return label;
}

std::string ClusterCommandProvider::prepareStatement(
    const std::span<const std::byte>& plan) {
  throw std::runtime_error("Unimplemented");
}

fs::path ClusterCommandProvider::runPreparedStatement(const std::string& label,
                                                      bool echo) {
  return fs::path();
}

fs::path ClusterCommandProvider::runStatement(const fs::path& plan, bool echo) {
  //  std::string label = p_impl->uue.inc_label();
  //  proteus::distributed::ClusterManager::getInstance().broadcastQuery(
  //      {label, std::make_unique<mmap_file>(plan, PAGEABLE)});

  // FIXME: how to get output-path of distributed query.
  return fs::path();
}
void ClusterCommandProvider::prepareStatement(
    const std::string& label, const std::span<const std::byte>& plan) {}
fs::path ClusterCommandProvider::runStatement(
    const std::span<const std::byte>& plan, bool echo) {
  return fs::path();
}

ClusterCommandProvider::ClusterCommandProvider() = default;
ClusterCommandProvider::~ClusterCommandProvider() = default;
