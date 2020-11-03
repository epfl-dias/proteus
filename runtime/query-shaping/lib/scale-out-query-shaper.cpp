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

#include "query-shaping/scale-out-query-shaper.hpp"

#include <olap/plan/catalog-parser.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>

namespace proteus {
ScaleOutQueryShaper::ScaleOutQueryShaper(std::string base_path)
    : InputPrefixQueryShaper(std::move(base_path)) {}

[[nodiscard]] pg ScaleOutQueryShaper::getPlugin() const {
  return pg{"distributed-block"};
}

[[nodiscard]] DeviceType ScaleOutQueryShaper::getDevice() {
  return DeviceType::CPU;
}

[[nodiscard]] int ScaleOutQueryShaper::getSlack() { return 128; }

[[nodiscard]] DegreeOfParallelism ScaleOutQueryShaper::getServerDOP() {
  return DegreeOfParallelism{2};
}

RelBuilder ScaleOutQueryShaper::scan(
    const std::string& relName, std::initializer_list<std::string> relAttrs) {
  return getBuilder().scan(getRelName(relName), relAttrs,
                           CatalogParser::getInstance(), getPlugin());
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::distribute_build(
    RelBuilder input) {
  auto rel = input
                 .membrdcst_scaleout(getServerDOP().operator unsigned long(),
                                     true, false)
                 .router_scaleout(
                     [&](const auto& arg) -> std::optional<expression_t> {
                       return arg["__broadcastTarget"];
                     },
                     getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED,
                     getDevice());

  return InputPrefixQueryShaper::distribute_build(rel);
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::distribute_probe(
    RelBuilder input) {
  auto rel = input
                 .router_scaleout(
                     [&](const auto& arg) -> std::optional<expression_t> {
                       return (int)(InfiniBandManager::server_id());
                     },
                     getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED,
                     getDevice())
                 .memmove_scaleout(getSlack());
  return InputPrefixQueryShaper::distribute_probe(rel);
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::collect_unpacked(
    RelBuilder input) {
  StorageManager::getInstance().unloadAll();
  return InputPrefixQueryShaper::collect_unpacked(input).router_scaleout(
      [&](const auto& arg) -> std::optional<expression_t> {
        return (int)0;  // std::nullopt;
      },
      getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice());
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::collect(RelBuilder input) {
  return collect_unpacked(input).memmove_scaleout(getSlackReduce());
}
}  // namespace proteus
