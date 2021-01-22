/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2021
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

#include <olap/plan/catalog-parser.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <query-shaping/experimental-shapers.hpp>

namespace proteus {

std::unique_ptr<Affinitizer> CPUOnlyShuffleAll::getAffinitizer() {
  return std::make_unique<CpuNumaNodeAffinitizer>();
}

[[nodiscard]] RelBuilder CPUOnlyShuffleAll::distribute_probe_interserver(
    RelBuilder input) {
  return input
      .router_scaleout(
          [&](const auto& arg) -> std::optional<expression_t> {
            return (int)(1 - InfiniBandManager::server_id());
          },
          getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice())
      .memmove_scaleout(getSlack());
}

[[nodiscard]] DeviceType GPUOnlyShuffleAll::getDevice() {
  return DeviceType::GPU;
}
[[nodiscard]] RelBuilder GPUOnlyShuffleAll::distribute_probe_interserver(
    RelBuilder input) {
  return input
      .router_scaleout(
          [&](const auto& arg) -> std::optional<expression_t> {
            return (int)(1 - InfiniBandManager::server_id());
          },
          getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice())
      .memmove_scaleout(getSlack());
}

[[nodiscard]] RelBuilder GPUOnlyShuffleAll::distribute_probe_intraserver(
    RelBuilder input) {
  auto rel = input.router(getDOP(), 2, RoutingPolicy::LOCAL, getDevice(),
                          getAffinitizer());

  if (doMove()) rel = rel.memmove(2, getDevice());

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

[[nodiscard]] RelBuilder GPUOnlyShuffleAll::collect(RelBuilder input) {
  return ScaleOutQueryShaper::collect(input).memmove(getSlackReduce(),
                                                     DeviceType::CPU);
}

std::unique_ptr<Affinitizer> CPUOnlyShuffleAllCorrectPlan::getAffinitizer() {
  return std::make_unique<CpuNumaNodeAffinitizer>();
}

[[nodiscard]] DeviceType CPUOnlyShuffleAllCorrectPlan::getDevice() {
  return DeviceType::CPU;
}

[[nodiscard]] RelBuilder
CPUOnlyShuffleAllCorrectPlan::distribute_probe_interserver(RelBuilder input) {
  return input.router_scaleout(
      [&](const auto& arg) -> std::optional<expression_t> {
        return (int)(1 - InfiniBandManager::server_id());
      },
      getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice());
}

[[nodiscard]] RelBuilder
CPUOnlyShuffleAllCorrectPlan::distribute_probe_intraserver(RelBuilder input) {
  auto rel = input
                 .router(getDOP(), 32, RoutingPolicy::RANDOM, getDevice(),
                         getAffinitizer())
                 .memmove_scaleout(256);

  //    if (doMove()) rel = rel.memmove(2, getDevice());

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

std::unique_ptr<Affinitizer> GPUOnlyShuffleAllCorrectPlan::getAffinitizer() {
  return std::make_unique<GPUAffinitizer>();
}

[[nodiscard]] DeviceType GPUOnlyShuffleAllCorrectPlan::getDevice() {
  return DeviceType::GPU;
}
[[nodiscard]] RelBuilder
GPUOnlyShuffleAllCorrectPlan::distribute_probe_interserver(RelBuilder input) {
  return input.router_scaleout(
      [&](const auto& arg) -> std::optional<expression_t> {
        return (int)(1 - InfiniBandManager::server_id());
      },
      getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice());
}

[[nodiscard]] RelBuilder
GPUOnlyShuffleAllCorrectPlan::distribute_probe_intraserver(RelBuilder input) {
  auto rel = input
                 .router(getDOP(), 32, RoutingPolicy::RANDOM, getDevice(),
                         getAffinitizer())
                 .memmove_scaleout(256);

  //    if (doMove()) rel = rel.memmove(2, getDevice());
  //                   .memmove_scaleout(1024, DeviceType::CPU);

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

[[nodiscard]] RelBuilder GPUOnlyShuffleAllCorrectPlan::collect(
    RelBuilder input) {
  return ScaleOutQueryShaper::collect(input).memmove(getSlackReduce(),
                                                     DeviceType::CPU);
}

[[nodiscard]] bool LazyGPUNoShuffle::doMove() { return false; }
[[nodiscard]] int LazyGPUNoShuffle::getSlack() { return 1024; }
[[nodiscard]] DeviceType LazyGPUNoShuffle::getDevice() {
  return DeviceType::GPU;
}

[[nodiscard]] RelBuilder LazyGPUNoShuffle::collect(RelBuilder input) {
  return ScaleOutQueryShaper::collect(input).memmove(getSlackReduce(),
                                                     DeviceType::CPU);
}

[[nodiscard]] RelBuilder
GPUOnlyLocalAllCorrectPlan::distribute_probe_interserver(RelBuilder input) {
  return input.router_scaleout(
      [&](const auto& arg) -> std::optional<expression_t> {
        return (int)(InfiniBandManager::server_id());
      },
      getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice());
}

[[nodiscard]] RelBuilder
GPUOnlyLocalAllCorrectPlan::distribute_probe_intraserver(RelBuilder input) {
  auto rel = input
                 .router(getDOP(), 32, RoutingPolicy::LOCAL, getDevice(),
                         getAffinitizer())
                 .memmove_scaleout(256);

  //    if (doMove()) rel = rel.memmove(2, getDevice());
  //                   .memmove_scaleout(1024, DeviceType::CPU);

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

[[nodiscard]] RelBuilder
LazyGPUOnlyLocalAllCorrectPlan::distribute_probe_intraserver(RelBuilder input) {
  auto rel = input.router(getDOP(), 8, RoutingPolicy::RANDOM, getDevice(),
                          getAffinitizer());  // FIXME

  //    if (doMove()) rel = rel.memmove(2, getDevice());
  //                   .memmove_scaleout(1024, DeviceType::CPU);

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

[[nodiscard]] bool LazyGPUOnlySingleSever::doMove() { return false; }
[[nodiscard]] int LazyGPUOnlySingleSever::getSlack() { return 256; }
[[nodiscard]] int LazyGPUOnlySingleSever::getSlackReduce() { return 32; }

[[nodiscard]] RelBuilder LazyGPUOnlySingleSever::collect(RelBuilder input) {
  return InputPrefixQueryShaper::collect(input).memmove(getSlackReduce(),
                                                        DeviceType::CPU);
}

[[nodiscard]] DeviceType CPUOnlySingleSever::getDevice() {
  return DeviceType::CPU;
}

std::unique_ptr<Affinitizer> CPUOnlySingleSever::getAffinitizer() {
  return std::make_unique<CpuNumaNodeAffinitizer>();
}

[[nodiscard]] RelBuilder GPUOnlyHalfFile::scan(
    const std::string& relName, std::initializer_list<std::string> relAttrs) {
  if (relName != "lineorder") {
    return proteus::InputPrefixQueryShaper::scan(relName, relAttrs);
  }
  auto rel =
      getBuilder().scan(getRelName(relName), relAttrs,
                        CatalogParser::getInstance(), pg{"distributed-block"});
  rel = rel.hintRowCount(getRowHint(relName) / 2);

  return rel;
}

[[nodiscard]] DeviceType CPUOnlyHalfFile::getDevice() {
  return DeviceType::CPU;
}

[[nodiscard]] std::unique_ptr<Affinitizer> CPUOnlyHalfFile::getAffinitizer() {
  return std::make_unique<CpuNumaNodeAffinitizer>();
}

}  // namespace proteus
