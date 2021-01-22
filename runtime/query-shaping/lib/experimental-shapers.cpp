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
            return (int)((InfiniBandManager::server_id() + 1) %
                         static_cast<size_t>(getServerDOP()));
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
            return (int)((InfiniBandManager::server_id() + 1) %
                         static_cast<size_t>(getServerDOP()));
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

[[nodiscard]] RelBuilder GPUOnlyShuffleAll::collect_packed(RelBuilder input) {
  return ScaleOutQueryShaper::collect_packed(input).memmove(getSlackReduce(),
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
        return (int)((InfiniBandManager::server_id() + 1) %
                     static_cast<size_t>(getServerDOP()));
      },
      getServerDOP(), getSlack(), RoutingPolicy::HASH_BASED, getDevice());
}

[[nodiscard]] RelBuilder
CPUOnlyShuffleAllCorrectPlan::distribute_probe_intraserver(RelBuilder input) {
  auto rel = input
                 .router(getDOP(), 2, RoutingPolicy::RANDOM, getDevice(),
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
        return (int)((InfiniBandManager::server_id() + 1) %
                     static_cast<size_t>(getServerDOP()));
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

[[nodiscard]] RelBuilder GPUOnlyShuffleAllCorrectPlan::collect_packed(
    RelBuilder input) {
  return ScaleOutQueryShaper::collect_packed(input).memmove(getSlackReduce(),
                                                            DeviceType::CPU);
}

[[nodiscard]] bool LazyGPUNoShuffle::doMove() { return false; }
[[nodiscard]] int LazyGPUNoShuffle::getSlack() { return 1024; }
[[nodiscard]] DeviceType LazyGPUNoShuffle::getDevice() {
  return DeviceType::GPU;
}

[[nodiscard]] RelBuilder LazyGPUNoShuffle::collect_packed(RelBuilder input) {
  return ScaleOutQueryShaper::collect_packed(input).memmove(getSlackReduce(),
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
  auto rel = input.router(getDOP(), 8, RoutingPolicy::LOCAL, getDevice(),
                          getAffinitizer());  // FIXME

  //    if (doMove()) rel = rel.memmove(2, getDevice());
  //                   .memmove_scaleout(1024, DeviceType::CPU);

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

[[nodiscard]] bool LazyGPUOnlySingleServer::doMove() { return false; }
[[nodiscard]] int LazyGPUOnlySingleServer::getSlack() { return 1024; }
[[nodiscard]] int LazyGPUOnlySingleServer::getSlackReduce() { return 32; }

[[nodiscard]] RelBuilder LazyGPUOnlySingleServer::collect_packed(
    RelBuilder input) {
  return InputPrefixQueryShaper::collect_packed(input).memmove(getSlackReduce(),
                                                               DeviceType::CPU);
}

[[nodiscard]] DeviceType CPUOnlySingleServer::getDevice() {
  return DeviceType::CPU;
}

std::unique_ptr<Affinitizer> CPUOnlySingleServer::getAffinitizer() {
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

RelBuilder HybridSingleServer::parallel(
    RelBuilder probe, const std::vector<RelBuilder>& builds,
    const std::function<RelBuilder(RelBuilder, const std::vector<RelBuilder>&)>&
        pathBuilder) {
  CPUOnlySingleServer morph_cpu{base_path, input_sizes, doMove(), 2};
  GPUOnlySingleServer morph_gpu{base_path, input_sizes, doMove(), 8};

  std::vector<RelBuilder> parallelizedBuilds_g;
  parallelizedBuilds_g.reserve(builds.size());
  std::vector<RelBuilder> parallelizedBuilds_c;
  parallelizedBuilds_c.reserve(builds.size());
  for (auto& build : builds) {
    auto pb = build.membrdcst(DegreeOfParallelism{2}, true, true)
                  .split(2, 8, RoutingPolicy::HASH_BASED);
    parallelizedBuilds_g.emplace_back(morph_gpu.distribute_build(pb));
    parallelizedBuilds_c.emplace_back(morph_cpu.distribute_build(pb));
  }

  auto parallelizedProbe = probe.split(2, 2, RoutingPolicy::RANDOM);

  auto rel_g = morph_gpu.distribute_probe(parallelizedProbe);
  auto rel_c = morph_cpu.distribute_probe(parallelizedProbe);

  auto rel_CPU = morph_cpu.collect(pathBuilder(rel_c, parallelizedBuilds_c));
  auto rel_GPU = morph_gpu.collect(pathBuilder(rel_g, parallelizedBuilds_g));

  return rel_GPU.unionAll({rel_CPU}).router(
      DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM, DeviceType::CPU);
}

RelBuilder LazyHybridSingleServer::parallel(
    RelBuilder probe, const std::vector<RelBuilder>& builds,
    const std::function<RelBuilder(RelBuilder, const std::vector<RelBuilder>&)>&
        pathBuilder) {
  CPUOnlySingleServer morph_cpu{base_path, input_sizes, doMove(), 2};
  LazyGPUOnlySingleServer morph_gpu{base_path, input_sizes, doMove(), 8};

  std::vector<RelBuilder> parallelizedBuilds_g;
  parallelizedBuilds_g.reserve(builds.size());
  std::vector<RelBuilder> parallelizedBuilds_c;
  parallelizedBuilds_c.reserve(builds.size());
  for (auto& build : builds) {
    auto pb = build.membrdcst(DegreeOfParallelism{2}, true, true)
                  .split(2, 8, RoutingPolicy::HASH_BASED);
    parallelizedBuilds_g.emplace_back(morph_gpu.distribute_build(pb));
    parallelizedBuilds_c.emplace_back(morph_cpu.distribute_build(pb));
  }

  auto parallelizedProbe = probe.split(2, 2, RoutingPolicy::RANDOM);

  auto rel_g = morph_gpu.distribute_probe(parallelizedProbe);
  auto rel_c = morph_cpu.distribute_probe(parallelizedProbe);

  auto rel_CPU = morph_cpu.collect(pathBuilder(rel_c, parallelizedBuilds_c));
  auto rel_GPU = morph_gpu.collect(pathBuilder(rel_g, parallelizedBuilds_g));

  return rel_GPU.unionAll({rel_CPU}).router(
      DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM, DeviceType::CPU);
}

RelBuilder HybridSingleServerMorsel::parallel(
    RelBuilder probe, const std::vector<RelBuilder>& builds,
    const std::function<RelBuilder(RelBuilder, const std::vector<RelBuilder>&)>&
        pathBuilder) {
  CPUOnlySingleServerMorsel morph_cpu{base_path, input_sizes, doMove(), 2};
  GPUOnlySingleServer morph_gpu{base_path, input_sizes, doMove(), 8};

  std::vector<RelBuilder> parallelizedBuilds_g;
  parallelizedBuilds_g.reserve(builds.size());
  std::vector<RelBuilder> parallelizedBuilds_c;
  parallelizedBuilds_c.reserve(builds.size());
  for (auto& build : builds) {
    auto pb = build.membrdcst(DegreeOfParallelism{2}, true, true)
                  .split(2, 8, RoutingPolicy::HASH_BASED);
    parallelizedBuilds_g.emplace_back(morph_gpu.distribute_build(pb));
    parallelizedBuilds_c.emplace_back(morph_cpu.distribute_build(pb));
  }

  auto parallelizedProbe = probe.split(2, 2, RoutingPolicy::RANDOM);

  auto rel_g = morph_gpu.distribute_probe(parallelizedProbe);
  auto rel_c = morph_cpu.distribute_probe(parallelizedProbe);

  auto rel_CPU = morph_cpu.collect(pathBuilder(rel_c, parallelizedBuilds_c));
  auto rel_GPU = morph_gpu.collect(pathBuilder(rel_g, parallelizedBuilds_g));

  return rel_GPU.unionAll({rel_CPU}).router(
      DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM, DeviceType::CPU);
}

RelBuilder LazyHybridSingleServerMorsel::parallel(
    RelBuilder probe, const std::vector<RelBuilder>& builds,
    const std::function<RelBuilder(RelBuilder, const std::vector<RelBuilder>&)>&
        pathBuilder) {
  CPUOnlySingleServerMorsel morph_cpu{base_path, input_sizes, doMove(), 2};
  LazyGPUOnlySingleServer morph_gpu{base_path, input_sizes, doMove(), 8};

  std::vector<RelBuilder> parallelizedBuilds_g;
  parallelizedBuilds_g.reserve(builds.size());
  std::vector<RelBuilder> parallelizedBuilds_c;
  parallelizedBuilds_c.reserve(builds.size());
  for (auto& build : builds) {
    auto pb = build.membrdcst(DegreeOfParallelism{2}, true, true)
                  .split(2, 8, RoutingPolicy::HASH_BASED);
    parallelizedBuilds_g.emplace_back(morph_gpu.distribute_build(pb));
    parallelizedBuilds_c.emplace_back(morph_cpu.distribute_build(pb));
  }

  auto parallelizedProbe = probe.split(2, 2, RoutingPolicy::RANDOM);

  auto rel_g = morph_gpu.distribute_probe(parallelizedProbe);
  auto rel_c = morph_cpu.distribute_probe(parallelizedProbe);

  auto rel_CPU = morph_cpu.collect(pathBuilder(rel_c, parallelizedBuilds_c));
  auto rel_GPU = morph_gpu.collect(pathBuilder(rel_g, parallelizedBuilds_g));

  return rel_GPU.unionAll({rel_CPU}).router(
      DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM, DeviceType::CPU);
}

RelBuilder CPUOnlySingleServerMorsel::distribute_build(RelBuilder input) {
  auto rel = input
                 .router(getDOP(), getSlack(), RoutingPolicy::LOCAL,
                         getDevice(), getAffinitizer())
                 .memmove(8, getDevice());

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}
}  // namespace proteus
