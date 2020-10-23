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

#include <unistd.h>

#include <cli-flags.hpp>
#include <command-provider/local-command-provider.hpp>
#include <map>
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

  void set_label(std::string label) { last_label = std::move(label); }

  void store(QueryResult &&qr) {
    last_result = std::make_unique<QueryResult>(std::move(qr));
  }

  void reset() { last_result.reset(); }
};

struct LocalCommandProvider::impl {
  unlink_upon_exit uue;
  std::map<std::string, PreparedStatement> preparedStatements;
};

std::string LocalCommandProvider::prepareStatement(const fs::path &plan) {
  mmap_file planfile{plan, PAGEABLE};
  return prepareStatement(planfile.asSpan());
}

std::string LocalCommandProvider::prepareStatement(
    const std::span<const std::byte> &plan) {
  std::string label = p_impl->uue.inc_label();
  p_impl->preparedStatements.emplace(
      label, PreparedStatement::from(plan, label, catalogJSON));

  return label;
}

void LocalCommandProvider::prepareStatement(
    const std::string &label, const std::span<const std::byte> &plan) {
  // std::string label = p_impl->uue.inc_label();
  p_impl->uue.set_label(label);
  p_impl->preparedStatements.emplace(
      label, PreparedStatement::from(plan, label, catalogJSON));
}

fs::path LocalCommandProvider::runStatement(const fs::path &plan, bool echo) {
  mmap_file planfile{plan, PAGEABLE};
  return runStatement(planfile.asSpan(), echo);
}

fs::path LocalCommandProvider::runStatement(
    const std::span<const std::byte> &plan, bool echo) {
  std::string label = p_impl->uue.inc_label();

  auto prepared = PreparedStatement::from(plan, label, catalogJSON);
  for (size_t i = 1; i < FLAGS_repeat; ++i) prepared.execute();
  auto qr = prepared.execute();

  if (echo) {
    std::cout << "result echo" << std::endl;
    std::cout << qr << std::endl;
  }

  p_impl->uue.store(std::move(qr));

  return fs::path{"/dev/shm"} / label;
}

fs::path LocalCommandProvider::runPreparedStatement(const std::string &label,
                                                    bool echo) {
  p_impl->uue
      .reset();  // Reset before exec to avoid conflicting with the output file
  auto &prepared = p_impl->preparedStatements.at(label);
  for (size_t i = 1; i < FLAGS_repeat; ++i) prepared.execute();
  auto qr = prepared.execute();

  if (echo) {
    std::cout << "result echo" << std::endl;
    std::cout << qr << std::endl;
  }

  p_impl->uue.store(std::move(qr));

  return fs::path{"/dev/shm"} / label;
}

LocalCommandProvider::LocalCommandProvider() = default;
LocalCommandProvider::~LocalCommandProvider() = default;
