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

#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <query-shaping/query-shaper.hpp>

namespace proteus {

RelBuilder QueryShaper::parallel(
    RelBuilder probe, const std::vector<RelBuilder> &builds,
    const std::function<RelBuilder(RelBuilder, const std::vector<RelBuilder> &)>
        &pathBuilder) {
  std::vector<RelBuilder> parallelizedBuilds;
  parallelizedBuilds.reserve(builds.size());
  for (auto &build : builds) {
    parallelizedBuilds.emplace_back(distribute_build(build));
  }

  auto parallelizedProbe = distribute_probe(probe);

  auto rel = pathBuilder(parallelizedProbe, parallelizedBuilds);

  return collect(rel);
}

RelBuilder QueryShaper::distribute_probe(RelBuilder input) {
  auto rel = input.router(getDOP(), getSlack(), RoutingPolicy::LOCAL,
                          getDevice(), getAffinitizer());

  if (doMove()) rel = rel.memmove(getSlack(), getDevice());

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

std::unique_ptr<Affinitizer> QueryShaper::getAffinitizer() {
  if (getDevice() == DeviceType::GPU) {
    return std::make_unique<GPUAffinitizer>();
  } else {
    return std::make_unique<CpuNumaNodeAffinitizer>();
  }
}

std::unique_ptr<Affinitizer> QueryShaper::getAffinitizerReduce() {
  return std::make_unique<CpuCoreAffinitizer>();
}

DeviceType QueryShaper::getDevice() { return DeviceType::GPU; }
int QueryShaper::getSlack() { return slack; }
int QueryShaper::getSlackReduce() { return 128; }
bool QueryShaper::doMove() { return true; }
size_t QueryShaper::sf() { return 100; }

double QueryShaper::StorageDOPFactor = 1.0;

DegreeOfParallelism QueryShaper::getDOP() {
  return DegreeOfParallelism{
      (getDevice() == DeviceType::CPU)
          ? (size_t)(topology::getInstance().getCoreCount() *
                     (InfiniBandManager::server_id() <
                              ((int)(InfiniBandManager::server_count() / 2))
                          ? 1
                          : StorageDOPFactor))
          : topology::getInstance().getGpuCount()};
}

RelBuilder QueryShaper::distribute_build(RelBuilder input) {
  auto rel =
      input
          .membrdcst(
              getDOP(), getDevice() == DeviceType::CPU,
              false)  //!(doMove() )) // || getDevice() == DeviceType::GPU
          .router(
              [&](const auto &arg) -> expression_t {
                return arg["__broadcastTarget"];
              },
              getDOP(), getSlack(), getDevice(), getAffinitizer());

  if (getDevice() == DeviceType::GPU) rel = rel.to_gpu();

  return rel;
}

RelBuilder QueryShaper::collect_unpacked(RelBuilder input) {
  if (getDevice() == DeviceType::GPU) input = input.to_cpu();
  return input.router(DegreeOfParallelism{1}, getSlackReduce(),
                      RoutingPolicy::RANDOM, DeviceType::CPU,
                      getAffinitizerReduce());
}

RelBuilder QueryShaper::collect_packed(RelBuilder input) {
  return collect_unpacked(input).memmove(getSlackReduce(), DeviceType::CPU);
}

RelBuilder QueryShaper::collect(RelBuilder input) {
  if (input.isPacked()) {
    return collect_packed(input);
  } else {
    return collect_unpacked(input);
  }
}

std::string QueryShaper::getRelName(const std::string &base) { return base; }

RelBuilder QueryShaper::scan(const std::string &relName,
                             std::initializer_list<std::string> relAttrs) {
  return getBuilder().scan(getRelName(relName), relAttrs,
                           CatalogParser::getInstance(), getPlugin());
}

pg QueryShaper::getPlugin() const { return pg{BinaryBlockPlugin::type}; }

RelBuilder QueryShaper::getBuilder() const { return ctx->getBuilder(); }

void QueryShaper::setQueryName(std::string name) {
  query = std::move(name);
  ctx = std::make_unique<RelBuilderFactory>(query);
}

QueryShaper::QueryShaper(size_t slack)
    : ctx(std::make_unique<RelBuilderFactory>("unnamed")), slack(slack) {}
QueryShaper::~QueryShaper() = default;

QueryShaperControlMoves::QueryShaperControlMoves(bool allow_moves, size_t slack)
    : QueryShaper(slack), allow_moves(allow_moves) {}

bool QueryShaperControlMoves::doMove() { return allow_moves; }

void QueryShaperControlMoves::setQueryName(std::string name) {
  QueryShaper::setQueryName(name + (doMove() ? "mv" : "nmv"));
}
}  // namespace proteus
