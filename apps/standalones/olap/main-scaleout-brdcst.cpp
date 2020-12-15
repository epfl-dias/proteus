/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#include <arpa/inet.h>
#include <gflags/gflags.h>
#include <rdma/rdma_cma.h>

#include <olap/operators/relbuilder-factory.hpp>
#include <olap/operators/relbuilder.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <olap/plan/prepared-statement.hpp>
#include <olap/plugins/binary-block-plugin.hpp>
#include <olap/util/parallel-context.hpp>
#include <platform/common/error-handling.hpp>
#include <platform/memory/block-manager.hpp>
#include <platform/memory/memory-manager.hpp>
#include <platform/network/infiniband/infiniband-manager.hpp>
#include <platform/storage/storage-manager.hpp>
#include <platform/topology/affinity_manager.hpp>
#include <platform/topology/topology.hpp>
#include <platform/util/logging.hpp>
#include <type_traits>

#include "cli-flags.hpp"

int main(int argc, char *argv[]) {
  auto ctx = proteus::from_cli::olap(
      "Simple command line interface for proteus", &argc, &argv);

  bool echo = false;

  // set_exec_location_on_scope affg{topology::getInstance().getGpus()[1]};
  set_exec_location_on_scope aff{
      topology::getInstance().getCpuNumaNodes()[FLAGS_primary ? 0 : 1]};

  assert(FLAGS_port <= std::numeric_limits<uint16_t>::max());
  InfiniBandManager::init(FLAGS_url, static_cast<uint16_t>(FLAGS_port),
                          FLAGS_primary, FLAGS_ipv4);

  auto ctx2 = new ParallelContext("main2", false);
  CatalogParser catalog2("inputs", ctx2);

  std::string d_datekey = "d_datekey";
  std::string d_year = "d_year";

  std::string lineorder = "inputs/ssbm100/lineorder.csv";

  std::string lo_quantity = "lo_quantity";
  std::string lo_discount = "lo_discount";
  std::string lo_extendedprice = "lo_extendedprice";
  std::string lo_orderdate = "lo_orderdate";

  // std::string lineorder = "inputs/ssbm100/date.csv";
  // std::string lo_orderdate = "d_datekey";

  // auto rel =
  //     RelBuilder{ctx}
  //         .scan<BinaryBlockPlugin>("inputs/ssbm1000/lineorder.csv",
  //                                    {lo_orderdate}, catalog)
  //         .router(
  //             [&](const auto &arg) -> std::vector<RecordAttribute *> {
  //               return {new RecordAttribute{
  //                   arg[lo_orderdate].getRegisteredAs(), false}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> {
  //               return std::nullopt;
  //             },
  //             DegreeOfParallelism{topology::getInstance().getGpuCount()},
  //             8, RoutingPolicy::LOCAL, DeviceType::GPU)
  //         .memmove(8, false)
  //         .to_gpu()
  //         .unpack([&](const auto &arg) -> std::vector<expression_t> {
  //           return {arg[lo_orderdate]};
  //         })
  //         .reduce(
  //             [&](const auto &arg) -> std::vector<expression_t> {
  //               return {arg[lo_orderdate]};
  //             },
  //             {SUM})
  //         .to_cpu()
  //         .router(
  //             [&](const auto &arg) -> std::vector<RecordAttribute *> {
  //               return {new RecordAttribute{
  //                   arg[lo_orderdate].getRegisteredAs(), false}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> {
  //               return std::nullopt;
  //             },
  //             DegreeOfParallelism{1}, 8, RoutingPolicy::RANDOM,
  //             DeviceType::CPU)
  //         .reduce(
  //             [&](const auto &arg) -> std::vector<expression_t> {
  //               return {arg[lo_orderdate]};
  //             },
  //             {SUM})
  //         .router_scaleout(
  //             [&](const auto &arg) -> std::vector<RecordAttribute *> {
  //               return {new RecordAttribute{
  //                   arg[lo_orderdate].getRegisteredAs(), false}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> {
  //               return std::nullopt;
  //             },
  //             1, 1, 8, RoutingPolicy::RANDOM, DeviceType::CPU, -1)
  //         .print([&](const auto &arg) -> std::vector<expression_t> {
  //           auto reg_as2 = new RecordAttribute(
  //               0, "t2", lo_orderdate,
  //               arg[lo_orderdate].getExpressionType());
  //           assert(reg_as2 && "Error registering expression as attribute");

  //           InputInfo *datasetInfo =
  //               catalog.getOrCreateInputInfo(reg_as2->getRelationName());
  //           datasetInfo->exprType = new BagType{
  //               RecordType{std::vector<RecordAttribute *>{reg_as2}}};

  //           return {arg[lo_orderdate].as(reg_as2)};
  //         });

  size_t slack = 8;
  // auto rel =
  //     RelBuilder{ctx}
  //         .scan<BinaryBlockPlugin>(
  //             lineorder,
  //             {lo_orderdate, lo_extendedprice, lo_discount, lo_quantity},
  //             catalog)
  //         .router_scaleout(1, 1, slack, RoutingPolicy::RANDOM,
  //                          DeviceType::CPU)
  //         .memmove_scaleout(slack)
  //         .unpack()
  //         .filter([&](const auto &arg) -> expression_t {
  //           return ge(arg[lo_discount], 1) & le(arg[lo_discount], 3) &
  //                  lt(arg[lo_quantity], 25);
  //         })
  //         .reduce(
  //             [&](const auto &arg) -> std::vector<expression_t> {
  //               return {(arg[lo_discount] * arg[lo_extendedprice] *
  //                        arg[lo_orderdate])
  //                           .as(lineorder, lo_orderdate)};
  //             },
  //             {SUM})
  //         .print([&](const auto &arg) -> std::vector<expression_t> {
  //           auto reg_as2 = new RecordAttribute(
  //               0, "t2", lo_orderdate,
  //               arg[lo_orderdate].getExpressionType());
  //           assert(reg_as2 && "Error registering expression as attribute");

  //           InputInfo *datasetInfo =
  //               catalog.getOrCreateInputInfo(reg_as2->getRelationName());
  //           datasetInfo->exprType = new BagType{
  //               RecordType{std::vector<RecordAttribute *>{reg_as2}}};

  //           return {arg[lo_orderdate].as(reg_as2)};
  //         });

  // auto rel =
  //     RelBuilder{ctx}
  //         .scan<BinaryBlockPlugin>(lineorder, {lo_orderdate}, catalog)
  //         .membrdcst_scaleout(2, false)
  //         .router_scaleout( [&](const auto &arg) ->
  //         std::vector<RecordAttribute *> {
  //               return {new RecordAttribute{
  //                   arg[lo_orderdate].getRegisteredAs(), false}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> {
  //               return std::nullopt;
  //             },
  //           2, 1, slack, RoutingPolicy::RANDOM,
  //                          DeviceType::CPU)
  //         // .memmove_scaleout(slack)
  //         .unpack()
  //         .reduce(
  //             [&](const auto &arg) -> std::vector<expression_t> {
  //               return {arg[lo_orderdate]};
  //             },
  //             {SUM})
  //         .print([&](const auto &arg) -> std::vector<expression_t> {
  //           auto reg_as2 = new RecordAttribute(
  //               0, "t2", lo_orderdate,
  //               arg[lo_orderdate].getExpressionType());
  //           assert(reg_as2 && "Error registering expression as attribute");

  //           InputInfo *datasetInfo =
  //               catalog.getOrCreateInputInfo(reg_as2->getRelationName());
  //           datasetInfo->exprType = new BagType{
  //               RecordType{std::vector<RecordAttribute *>{reg_as2}}};

  //           return {arg[lo_orderdate].as(reg_as2)};
  //         });

  // auto rel =
  //     RelBuilder{ctx}
  //         .scan<BinaryBlockPlugin>(lineorder, {lo_orderdate}, catalog)
  //         .membrdcst_scaleout(2, false)
  //         .router_scaleout(
  //             [&](const auto &arg) -> std::vector<RecordAttribute *> {
  //               return {
  //                   new
  //                   RecordAttribute{arg[lo_orderdate].getRegisteredAs()}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> {
  //               return arg["__broadcastTarget"];
  //             },
  //             2, 1, slack, RoutingPolicy::HASH_BASED, DeviceType::CPU)
  //         // .memmove_scaleout(slack)
  //         .unpack()
  //         .reduce(
  //             [&](const auto &arg) -> std::vector<expression_t> {
  //               return {arg[lo_orderdate]};
  //             },
  //             {SUM})
  //         .print([&](const auto &arg) -> std::vector<expression_t> {
  //           auto reg_as2 = new RecordAttribute(
  //               0, "t2", lo_orderdate,
  //               arg[lo_orderdate].getExpressionType());
  //           assert(reg_as2 && "Error registering expression as attribute");

  //           InputInfo *datasetInfo =
  //               catalog.getOrCreateInputInfo(reg_as2->getRelationName());
  //           datasetInfo->exprType = new BagType{
  //               RecordType{std::vector<RecordAttribute *>{reg_as2}}};

  //           return {arg[lo_orderdate].as(reg_as2)};
  //         });
  auto rel =
      RelBuilderFactory{"main"}
          .getBuilder()
          .scan(lineorder, {lo_orderdate}, catalog2,
                pg{BinaryBlockPlugin::type})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[lo_orderdate].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return std::nullopt;
              },
              DegreeOfParallelism{2}, slack, RoutingPolicy::RANDOM,
              DeviceType::CPU)
          .memmove_scaleout(slack)
          .unpack()
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[lo_orderdate]};
              },
              {SUM})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[lo_orderdate].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> { return 0; },

              DegreeOfParallelism{1}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[lo_orderdate]};
              },
              {SUM})
          .print(pg{"pm-csv"});

  auto statement = rel.prepare();

  for (int i = 0; i < 1; ++i) {
    auto &sub = InfiniBandManager::subscribe();
    if (FLAGS_primary) {
      std::cout << "send" << std::endl;
      void *ptr = BlockManager::get_buffer();
      ((int *)ptr)[0] = 45;
      InfiniBandManager::send(ptr, 4);
      std::cout << "send done" << std::endl;
    } else {
      sleep(2);
      std::cout << "wait" << std::endl;
      sub.wait();
      std::cout << "wait done" << std::endl;
      //      auto v = sub.wait();
      //      BlockManager::release_buffer((int32_t *) v.data);
    }

    if (FLAGS_primary) {
      sub.wait();
      //      auto v = sub.wait();
      //      BlockManager::release_buffer((int32_t *) v.data);
    } else {
      std::cout << "send" << std::endl;
      void *ptr = BlockManager::get_buffer();
      ((int *)ptr)[0] = 44;
      InfiniBandManager::send(ptr, 4);
      std::cout << "send done" << std::endl;
    }
  }

  for (size_t i = 0; i < FLAGS_repeat; ++i) {
    auto qr = statement.execute();
    LOG(INFO) << "start of result";
    LOG(INFO) << qr;
    LOG(INFO) << "end of result";
  }

  if (FLAGS_primary) {
  } else {
    InfiniBandManager::disconnectAll();
  }
  StorageManager::getInstance().unloadAll();
  InfiniBandManager::deinit();
  return 0;
}
