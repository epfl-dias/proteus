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

#ifndef PROTEUS_NDP_HPP
#define PROTEUS_NDP_HPP

#include <command-provider/command-provider.hpp>
#include <distributed-runtime/cluster-manager.hpp>
#include <memory>
#include <plan-parser/plan-parser.hpp>
#include <span>
#include <threadpool/threadpool.hpp>
#include <utility>

namespace proteus {

class CommandExecutor {
 public:
  QueryResult run(PreparedStatement &statement) { return statement.execute(); }
};

class [[nodiscard]] ndp {
 private:
  class impl;
  const std::unique_ptr<impl> p_impl{};

 public:
  explicit ndp(const string &usage, int *argc, char ***argv);
  ~ndp();

  [[nodiscard]] proteus::distributed::ClusterManager &getClusterManager() const;
  ThreadPool &getThreadPool();
  PlanParser &getPlanParser();
  CommandProvider &getCommandProvider();
  CommandExecutor &getExecutor();
};
}  // namespace proteus

namespace proteus::from_cli {
proteus::ndp ndp(const std::string &usage, int *argc, char ***argv);
}  // namespace proteus::from_cli

#endif /* PROTEUS_NDP_HPP */
