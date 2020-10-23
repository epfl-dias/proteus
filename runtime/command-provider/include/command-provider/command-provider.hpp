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

#ifndef PROTEUS_COMMAND_PROVIDER_HPP
#define PROTEUS_COMMAND_PROVIDER_HPP

#include <span>
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#else
// TODO: remove as soon as the default GCC moves filesystem out of experimental
//  GCC 8.3 has made the transition, but the default GCC in Ubuntu 18.04 is 7.4
#include <experimental/filesystem>
namespace std {
namespace filesystem = std::experimental::filesystem;
}
#endif

namespace proteus {
class shutdown_command : public std::exception {};

class unprepared_plan_execution : public std::exception {};

class query_interrupt_command : public std::exception {};
}  // namespace proteus

class CommandProvider {
 public:
  /**
   * Prepare statement from JSON byte-stream
   *
   * @param plan NULL-terminated byte-stream containing the JSON-serialized plan
   * @param label identifying the prepared statement
   */
  virtual void prepareStatement(const std::string &label,
                                const std::span<const std::byte> &plan) = 0;

  /**
   * Prepare statement from JSON byte-stream
   *
   * @param plan NULL-terminated byte-stream containing the JSON-serialized plan
   * @returns label identifying the prepared statement
   */
  virtual std::string prepareStatement(
      const std::span<const std::byte> &plan) = 0;

  /**
   * Runs statement from JSON-serialized byte-stream plan
   * The returned value will only be valid for plans that return data in the
   * current server.
   *
   * @param plan JSON plan
   * @param echo If true, the query output is also printed on LOG(INFO)
   * @return Path pointing to the root folder of the results (plugin-specific)
   */
  virtual fs::path runStatement(const std::span<const std::byte> &plan,
                                bool echo = true) = 0;
  /**
   * Runs prepared statement identified by @p label label.
   * The returned value will only be valid for plans that return data in the
   * current server.
   *
   * @param label Prepared statement label
   * @param echo If true, the query output is also printed on LOG(INFO)
   * @return Path pointing to the root folder of the results (plugin-specific)
   */
  virtual fs::path runPreparedStatement(const std::string &label,
                                        bool echo = true) = 0;

  /**
   * Prepare statement from JSON file @p plan
   *
   * @param plan JSON-serialized plan
   * @returns label identifying the prepared statement
   */
  virtual std::string prepareStatement(const fs::path &plan) = 0;

  /**
   * Runs statement from JSON file
   * The returned value will only be valid for plans that return data in the
   * current server.
   *
   * @param plan path to JSON plan
   * @param echo If true, the query output is also printed on LOG(INFO)
   * @return Path pointing to the root folder of the results (plugin-specific)
   */
  virtual fs::path runStatement(const fs::path &plan, bool echo = true) = 0;

  virtual ~CommandProvider() = default;
};

#endif /* PROTEUS_COMMAND_PROVIDER_HPP */
