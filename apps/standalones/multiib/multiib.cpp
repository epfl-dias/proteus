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

#include <chrono>
#include <cli-flags.hpp>
#include <network/infiniband/infiniband-manager.hpp>
#include <olap/operators/relbuilder-factory.hpp>
#include <olap/operators/relbuilder.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <storage/storage-manager.hpp>
#include <topology/affinity_manager.hpp>
#include <topology/topology.hpp>

static std::string date = "inputs/ssbm1000/date.csv";

static std::string d_datekey = "d_datekey";
static std::string d_year = "d_year";
static std::string d_weeknuminyear = "d_weeknuminyear";

static std::string lineorder = "inputs/ssbm1000/lineorder.csv";

static std::string lo_quantity = "lo_quantity";
static std::string lo_discount = "lo_discount";
static std::string lo_extendedprice = "lo_extendedprice";
static std::string lo_orderdate = "lo_orderdate";

static std::string revenue = "revenue";

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap("Multi IB", &argc, &argv);

  set_exec_location_on_scope exec(topology::getInstance().getCpuNumaNodes()[0]);

  assert(FLAGS_port <= std::numeric_limits<uint16_t>::max());
  InfiniBandManager::init(FLAGS_url, static_cast<uint16_t>(FLAGS_port),
                          FLAGS_primary, FLAGS_ipv4);
  std::vector<PreparedStatement> rel;
  rel.reserve(5);
  for (size_t i = 0; i < 1; ++i) {
    rel.push_back(
        RelBuilderFactory{__FUNCTION__ + std::to_string(i)}
            .getBuilder()
            .scan("inputs/ssbm1000/lineorder.csv", {lo_orderdate},
                  CatalogParser::getInstance(), pg{"distributed-block"})
            .router_scaleout(
                [&](const auto &arg) -> std::optional<expression_t> {
                  return (int)(1 - InfiniBandManager::server_id());
                },
                DegreeOfParallelism{2}, 8, RoutingPolicy::HASH_BASED,
                DeviceType::CPU, 2)
            .memmove_scaleout(8)
            .router(DegreeOfParallelism{topology::getInstance().getCoreCount()},
                    8, RoutingPolicy::LOCAL, DeviceType::CPU)
            .memmove(8, DeviceType::CPU)
            //          .to_gpu()
            .unpack()
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[lo_orderdate]};
                },
                {SUM})
            //          .to_cpu()
            .router(DegreeOfParallelism{1}, 8, RoutingPolicy::RANDOM,
                    DeviceType::CPU)
            .router_scaleout(
                [&](const auto &arg) -> std::optional<expression_t> {
                  return (int)0;  // std::nullopt;
                },
                DegreeOfParallelism{2}, 8, RoutingPolicy::HASH_BASED,
                DeviceType::CPU, 2)
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[lo_orderdate]};
                },
                {SUM})
            .print(pg{"pm-csv"})
            .prepare());
  }

  using namespace std::chrono_literals;
  std::this_thread::sleep_for(5s);

  for (size_t i = 0; i < 5; ++i) {
    LOG(INFO) << rel[0].execute();
    std::this_thread::sleep_for(5s);
  }

  if (!FLAGS_primary) InfiniBandManager::disconnectAll();

  StorageManager::getInstance().unloadAll();
  InfiniBandManager::deinit();

  return 0;
}
