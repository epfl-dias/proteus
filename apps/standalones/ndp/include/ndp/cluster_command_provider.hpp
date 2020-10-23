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

#ifndef PROTEUS_CLUSTER_COMMAND_PROVIDER_HPP
#define PROTEUS_CLUSTER_COMMAND_PROVIDER_HPP

#include <command-provider/command-provider.hpp>
#include <memory>

/*
 * Receives commands from the query optimizer and spawns them over the cluster
 */
class ClusterCommandProvider : public CommandProvider {
 private:
  struct impl;
  std::unique_ptr<impl> p_impl;

 public:
  /**
   * Prepare statement from JSON byte-stream
   *
   * @param plan NULL-terminated byte-stream containing the JSON-serialized plan
   * @param label identifying the prepared statement
   */
  void prepareStatement(const std::string &label,
                        const std::span<const std::byte> &plan) override;

  /**
   * Prepare statement from JSON byte-stream
   *
   * @param plan NULL-terminated byte-stream containing the JSON-serialized plan
   * @returns label identifying the prepared statement
   */
  std::string prepareStatement(const std::span<const std::byte> &plan) override;

  /**
   * Runs statement from JSON-serialized byte-stream plan
   * The returned value will only be valid for plans that return data in the
   * current server.
   *
   * @param plan JSON plan
   * @param echo If true, the query output is also printed on LOG(INFO)
   * @return Path pointing to the root folder of the results (plugin-specific)
   */
  fs::path runStatement(const std::span<const std::byte> &plan,
                        bool echo = false) override;
  /**
   * Runs prepared statement identified by @p label label.
   * The returned value will only be valid for plans that return data in the
   * current server.
   *
   * @param label Prepared statement label
   * @param echo If true, the query output is also printed on LOG(INFO)
   * @return Path pointing to the root folder of the results (plugin-specific)
   */
  fs::path runPreparedStatement(const std::string &label,
                                bool echo = false) override;

  /**
   * Prepare statement from JSON file @p plan
   *
   * @param plan JSON-serialized plan
   * @returns label identifying the prepared statement
   */
  std::string prepareStatement(const fs::path &plan) override;

  /**
   * Runs statement from JSON file
   * The returned value will only be valid for plans that return data in the
   * current server.
   *
   * @param plan path to JSON plan
   * @param echo If true, the query output is also printed on LOG(INFO)
   * @return Path pointing to the root folder of the results (plugin-specific)
   */
  fs::path runStatement(const fs::path &plan, bool echo = false) override;

  ClusterCommandProvider();
  ~ClusterCommandProvider() override;
};

#endif /* PROTEUS_CLUSTER_COMMAND_PROVIDER_HPP */
