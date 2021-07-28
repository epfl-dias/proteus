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

#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>

namespace proteus {
ScaleOutQueryShaper::ScaleOutQueryShaper(std::string base_path,
                                         decltype(input_sizes) input_sizes)
    : InputPrefixQueryShaper(std::move(base_path), std::move(input_sizes)) {}

[[nodiscard]] pg ScaleOutQueryShaper::getPlugin() const {
  return pg{"registry-based-block"};
}

[[nodiscard]] DeviceType ScaleOutQueryShaper::getDevice() {
  return DeviceType::CPU;
}

[[nodiscard]] int ScaleOutQueryShaper::getSlack() { return 128; }

[[nodiscard]] DegreeOfParallelism ScaleOutQueryShaper::getServerDOP() {
  return DegreeOfParallelism{InfiniBandManager::server_count()};
}

RelBuilder ScaleOutQueryShaper::getBuilder() const {
  static RelBuilderFactory ctx{query + "distributed"};
  return ctx.getBuilder();
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

[[nodiscard]] RelBuilder ScaleOutQueryShaper::distribute_probe_interserver(
    RelBuilder input) {
  return input
      .router_scaleout(
          [&](const auto& arg) -> std::optional<expression_t> {
            return (int)InfiniBandManager::server_id();
          },
          getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice())
      .memmove_scaleout(getSlack());
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::distribute_probe_intraserver(
    RelBuilder input) {
  auto rel = input.router(getDOP(), 2, RoutingPolicy::LOCAL, getDevice(),
                          getAffinitizer());

  //  if (doMove()) rel = rel.memmove(2, getDevice());

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
  //  return InputPrefixQueryShaper::distribute_probe(rel);
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::distribute_probe(
    RelBuilder input) {
  return distribute_probe_intraserver(distribute_probe_interserver(input));
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::collect_unpacked(
    RelBuilder input) {
  return InputPrefixQueryShaper::collect_unpacked(input).router_scaleout(
      [&](const auto& arg) -> std::optional<expression_t> {
        return (int)0;  // std::nullopt;
      },
      getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice());
}

[[nodiscard]] RelBuilder ScaleOutQueryShaper::collect_packed(RelBuilder input) {
  return collect_unpacked(input).memmove_scaleout(getSlackReduce());
}

double ScaleOutQueryShaper::getRowHint(const std::string& relName) {
  auto rowhint = InputPrefixQueryShaper::getRowHint(relName);
  // if table size is too small to break across servers, it won't so
  // let's use the block size as a type-invariant heuristic, until
  // we have a proper interface to ask the plugins or the catalog
  if (rowhint <= BlockManager::block_size) return rowhint;
  return rowhint / getServerDOP();
}
}  // namespace proteus
