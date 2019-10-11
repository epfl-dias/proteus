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
#include <err.h>
#include <gflags/gflags.h>
#include <netdb.h>
#include <rdma/rdma_cma.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <network/infiniband/infiniband-manager.hpp>
#include <operators/relbuilder-factory.hpp>
#include <type_traits>

#include "cli-flags.hpp"
#include "common/error-handling.hpp"
#include "memory/block-manager.hpp"
#include "memory/memory-manager.hpp"
#include "operators/relbuilder.hpp"
#include "plan/catalog-parser.hpp"
#include "plan/prepared-statement.hpp"
#include "plugins/binary-block-plugin.hpp"
#include "storage/storage-manager.hpp"
#include "topology/affinity_manager.hpp"
#include "topology/topology.hpp"
#include "util/logging.hpp"
#include "util/parallel-context.hpp"
#include "util/profiling.hpp"

std::string date = "inputs/ssbm1000/date.csv";

std::string d_datekey = "d_datekey";
std::string d_year = "d_year";
std::string d_weeknuminyear = "d_weeknuminyear";

std::string lineorder = "inputs/ssbm1000/lineorder.csv";

std::string lo_quantity = "lo_quantity";
std::string lo_discount = "lo_discount";
std::string lo_extendedprice = "lo_extendedprice";
std::string lo_orderdate = "lo_orderdate";

std::string revenue = "revenue";

auto scan_sum(ParallelContext *ctx, CatalogParser &catalog, size_t slack) {
  auto rel =
      RelBuilderFactory{__FUNCTION__}
          .getBuilder()
          .scan<BinaryBlockPlugin>(
              lineorder,
              // {lo_orderdate, lo_quantity, lo_discount, lo_extendedprice},
              {lo_discount}, catalog)
          .unpack()
          // .filter([&](const auto &arg) -> expression_t {
          //   return lt(arg[lo_quantity], 25) & ge(arg[lo_discount], 1) &
          //          le(arg[lo_discount], 3);
          // })
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[lo_discount].as(lineorder, revenue)};
              },
              {SUM})
          .print(pg{"pm-csv"});

  return rel.prepare();
}

auto scan_sum_scaleout(ParallelContext *ctx, CatalogParser &catalog,
                       size_t slack) {
  size_t dop = topology::getInstance().getCoreCount();
  if (dop > 100) {
    // dop /= 4;
  } else {
    // dop /= 2;
  }
  auto rel =
      RelBuilderFactory{__FUNCTION__}
          .getBuilder()
          .scan<BinaryBlockPlugin>(lineorder, {lo_orderdate, lo_discount},
                                   catalog)
          .router(DegreeOfParallelism{dop}, slack, RoutingPolicy::LOCAL,
                  DeviceType::CPU)
          .unpack()
          // .filter([&](const auto &arg) -> expression_t {
          //   return lt(arg[lo_quantity], 25) & ge(arg[lo_discount], 1) &
          //          le(arg[lo_discount], 3);
          // })
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {(arg[lo_discount] + arg[lo_orderdate])
                            .as(lineorder, revenue)};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, slack, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .print(pg{"pm-csv"});

  return rel.prepare();
}

auto ssb_q11(ParallelContext *ctx, CatalogParser &catalog, size_t slack) {
  RelBuilderFactory factory{__FUNCTION__};
  auto rel_date =
      factory.getBuilder()
          .scan<BinaryBlockPlugin>(date, {d_datekey, d_year}, catalog)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg[d_year], 1993);
          });

  auto rel = factory.getBuilder()
                 .scan<BinaryBlockPlugin>(
                     lineorder,
                     {lo_orderdate, lo_quantity, lo_discount, lo_extendedprice},
                     catalog)
                 .filter([&](const auto &arg) -> expression_t {
                   return lt(arg[lo_quantity], 25) & ge(arg[lo_discount], 1) &
                          le(arg[lo_discount], 3);
                 })
                 .join(
                     rel_date,
                     [&](const auto &build_arg) -> expression_t {
                       return build_arg[d_datekey];
                     },
                     [&](const auto &build_arg) -> std::vector<GpuMatExpr> {
                       return {};
                     },
                     {64},
                     [&](const auto &probe_arg) -> expression_t {
                       return probe_arg[lo_orderdate];
                     },
                     [&](const auto &probe_arg) -> std::vector<GpuMatExpr> {
                       return {GpuMatExpr{(probe_arg[lo_discount] *
                                           probe_arg[lo_extendedprice])
                                              .as(lineorder, revenue),
                                          1, 0}};
                     },
                     {64, 32}, 9, 1024)
                 .reduce(
                     [&](const auto &arg) -> std::vector<expression_t> {
                       return {arg[revenue]};
                     },
                     {SUM})
                 .print(pg{"pm-csv"});

  return rel.prepare();
}

auto ssb_q11_par(ParallelContext *ctx, CatalogParser &catalog, size_t slack) {
  RelBuilderFactory factory{__FUNCTION__};
  size_t dop = topology::getInstance().getCoreCount();
  if (dop > 100) {
    dop /= 2;
  } else {
    // dop /= 2;
  }
  auto rel_date =
      factory.getBuilder()
          .scan<BinaryBlockPlugin>(date, {d_datekey, d_year}, catalog)
          .membrdcst(DegreeOfParallelism{dop}, true)
          .router(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {new RecordAttribute{arg[d_datekey].getRegisteredAs()},
                        new RecordAttribute{arg[d_year].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              DegreeOfParallelism{dop}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg[d_year], 1993);
          });

  auto rel = factory.getBuilder()
                 .scan<BinaryBlockPlugin>(
                     lineorder,
                     {lo_orderdate, lo_quantity, lo_discount, lo_extendedprice},
                     catalog)
                 .router(DegreeOfParallelism{dop}, slack, RoutingPolicy::LOCAL,
                         DeviceType::CPU)
                 .unpack()
                 .filter([&](const auto &arg) -> expression_t {
                   return lt(arg[lo_quantity], 25) & ge(arg[lo_discount], 1) &
                          le(arg[lo_discount], 3);
                 })
                 .join(
                     rel_date,
                     [&](const auto &build_arg) -> expression_t {
                       return build_arg[d_datekey];
                     },
                     [&](const auto &build_arg) -> std::vector<GpuMatExpr> {
                       return {};
                     },
                     {64},
                     [&](const auto &probe_arg) -> expression_t {
                       return probe_arg[lo_orderdate];
                     },
                     [&](const auto &probe_arg) -> std::vector<GpuMatExpr> {
                       return {GpuMatExpr{(probe_arg[lo_discount] *
                                           probe_arg[lo_extendedprice])
                                              .as(lineorder, revenue),
                                          1, 0}};
                     },
                     {64, 32}, 9, 1024)
                 .reduce(
                     [&](const auto &arg) -> std::vector<expression_t> {
                       return {arg[revenue]};
                     },
                     {SUM})
                 .router(DegreeOfParallelism{1}, slack, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto &arg) -> std::vector<expression_t> {
                       return {arg[revenue]};
                     },
                     {SUM})
                 .print(pg{"pm-csv"});

  return rel.prepare();
}

auto ssb_q13_par(ParallelContext *ctx, CatalogParser &catalog, size_t slack) {
  RelBuilderFactory factory{__FUNCTION__};
  size_t dop = topology::getInstance().getCoreCount();
  if (dop > 100) {
    dop /= 2;
  } else {
    // dop /= 2;
  }
  auto rel_date =
      factory.getBuilder()
          .scan<BinaryBlockPlugin>(date, {d_datekey, d_weeknuminyear, d_year},
                                   catalog)
          .membrdcst(DegreeOfParallelism{dop}, true)
          .router(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[d_datekey].getRegisteredAs()},
                    new RecordAttribute{arg[d_weeknuminyear].getRegisteredAs()},
                    new RecordAttribute{arg[d_year].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              DegreeOfParallelism{dop}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg[d_weeknuminyear], 6) & eq(arg[d_year], 1994);
          });

  auto rel = factory.getBuilder()
                 .scan<BinaryBlockPlugin>(
                     lineorder,
                     {lo_orderdate, lo_quantity, lo_discount, lo_extendedprice},
                     catalog)
                 .router(DegreeOfParallelism{dop}, slack, RoutingPolicy::LOCAL,
                         DeviceType::CPU)
                 .unpack()
                 .filter([&](const auto &arg) -> expression_t {
                   return ge(arg[lo_discount], 5) & le(arg[lo_discount], 7) &
                          ge(arg[lo_quantity], 26) & le(arg[lo_quantity], 35);
                 })
                 .join(
                     rel_date,
                     [&](const auto &build_arg) -> expression_t {
                       return build_arg[d_datekey];
                     },
                     [&](const auto &build_arg) -> std::vector<GpuMatExpr> {
                       return {};
                     },
                     {64},
                     [&](const auto &probe_arg) -> expression_t {
                       return probe_arg[lo_orderdate];
                     },
                     [&](const auto &probe_arg) -> std::vector<GpuMatExpr> {
                       return {GpuMatExpr{(probe_arg[lo_discount] *
                                           probe_arg[lo_extendedprice])
                                              .as(lineorder, revenue),
                                          1, 0}};
                     },
                     {64, 32}, 4, 16)
                 .reduce(
                     [&](const auto &arg) -> std::vector<expression_t> {
                       return {arg[revenue]};
                     },
                     {SUM})
                 .router(DegreeOfParallelism{1}, slack, RoutingPolicy::RANDOM,
                         DeviceType::CPU)
                 .reduce(
                     [&](const auto &arg) -> std::vector<expression_t> {
                       return {arg[revenue]};
                     },
                     {SUM})
                 .print(pg{"pm-csv"});

  return rel.prepare();
}

auto ssb_q11_scaleout(ParallelContext *ctx, CatalogParser &catalog,
                      size_t slack) {
  RelBuilderFactory factory{__FUNCTION__};
  auto rel_date =
      factory.getBuilder()
          .scan(date, {d_datekey, d_year}, catalog, pg{"distributed-block"})
          .membrdcst_scaleout(2, false)
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {new RecordAttribute{arg[d_datekey].getRegisteredAs()},
                        new RecordAttribute{arg[d_year].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              DegreeOfParallelism{2}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg[d_year], 1993);
          });

  auto rel =
      factory.getBuilder()
          .scan(lineorder,
                {lo_orderdate, lo_quantity, lo_discount, lo_extendedprice},
                catalog, pg{"distributed-block"})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[lo_orderdate].getRegisteredAs()},
                    new RecordAttribute{arg[lo_quantity].getRegisteredAs()},
                    new RecordAttribute{arg[lo_discount].getRegisteredAs()},
                    new RecordAttribute{
                        arg[lo_extendedprice].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return std::nullopt;
              },
              DegreeOfParallelism{2}, slack, RoutingPolicy::RANDOM,
              DeviceType::CPU, 1)
          .memmove_scaleout(slack)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return lt(arg[lo_quantity], 25) & ge(arg[lo_discount], 1) &
                   le(arg[lo_discount], 3);
          })
          .join(
              rel_date,
              [&](const auto &build_arg) -> expression_t {
                return build_arg[d_datekey];
              },
              [&](const auto &build_arg) -> std::vector<GpuMatExpr> {
                return {};
              },
              {64},
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg[lo_orderdate];
              },
              [&](const auto &probe_arg) -> std::vector<GpuMatExpr> {
                return {GpuMatExpr{
                    (probe_arg[lo_discount] * probe_arg[lo_extendedprice])
                        .as(lineorder, revenue),
                    1, 0}};
              },
              {64, 32}, 9, 1024)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {new RecordAttribute{arg[revenue].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> { return 0; },
              DegreeOfParallelism{1}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .print(pg{"pm-csv"});

  return rel.prepare();
}

auto ssb_q11_par_scaleout(ParallelContext *ctx, CatalogParser &catalog,
                          size_t slack) {
  RelBuilderFactory factory{__FUNCTION__};
  size_t dop = topology::getInstance().getCoreCount();
  if (dop > 100) {
    dop /= 2;
  } else {
    // dop /= 2;
  }
  auto rel_date =
      factory.getBuilder()
          .scan(date, {d_datekey, d_year}, catalog, pg{"distributed-block"})
          .membrdcst_scaleout(2, false)
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {new RecordAttribute{arg[d_datekey].getRegisteredAs()},
                        new RecordAttribute{arg[d_year].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              DegreeOfParallelism{2}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .membrdcst(DegreeOfParallelism{dop}, true)
          .router(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {new RecordAttribute{arg[d_datekey].getRegisteredAs()},
                        new RecordAttribute{arg[d_year].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              DegreeOfParallelism{dop}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg[d_year], 1993);
          });

  auto rel =
      factory.getBuilder()
          .scan(lineorder,
                {lo_orderdate, lo_quantity, lo_discount, lo_extendedprice},
                catalog, pg{"distributed-block"})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[lo_orderdate].getRegisteredAs()},
                    new RecordAttribute{arg[lo_quantity].getRegisteredAs()},
                    new RecordAttribute{arg[lo_discount].getRegisteredAs()},
                    new RecordAttribute{
                        arg[lo_extendedprice].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return (int)InfiniBandManager::server_id();  // std::nullopt;
              },
              DegreeOfParallelism{2}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .memmove_scaleout(16 * 1024)
          .router(DegreeOfParallelism{dop}, slack, RoutingPolicy::LOCAL,
                  DeviceType::CPU)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return lt(arg[lo_quantity], 25) & ge(arg[lo_discount], 1) &
                   le(arg[lo_discount], 3);
          })
          .join(
              rel_date,
              [&](const auto &build_arg) -> expression_t {
                return build_arg[d_datekey];
              },
              [&](const auto &build_arg) -> std::vector<GpuMatExpr> {
                return {};
              },
              {64},
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg[lo_orderdate];
              },
              [&](const auto &probe_arg) -> std::vector<GpuMatExpr> {
                return {GpuMatExpr{
                    (probe_arg[lo_discount] * probe_arg[lo_extendedprice])
                        .as(lineorder, revenue),
                    1, 0}};
              },
              {64, 32}, 9, 1024)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, slack, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {new RecordAttribute{arg[revenue].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> { return 0; },
              DegreeOfParallelism{1}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .print(pg{"pm-csv"});

  return rel.prepare();
}

auto ssb_q13_par_scaleout(ParallelContext *ctx, CatalogParser &catalog,
                          size_t slack) {
  RelBuilderFactory factory{__FUNCTION__};
  size_t dop = topology::getInstance().getCoreCount();
  if (dop > 100) {
    dop /= 2;
  } else {
    // dop /= 2;
  }

  auto rel_date =
      factory.getBuilder()
          .scan(date, {d_datekey, d_weeknuminyear, d_year}, catalog,
                pg{"distributed-block"})
          .membrdcst_scaleout(2, false)
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[d_datekey].getRegisteredAs()},
                    new RecordAttribute{arg[d_weeknuminyear].getRegisteredAs()},
                    new RecordAttribute{arg[d_year].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              DegreeOfParallelism{2}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .membrdcst(DegreeOfParallelism{dop}, true)
          .router(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[d_datekey].getRegisteredAs()},
                    new RecordAttribute{arg[d_weeknuminyear].getRegisteredAs()},
                    new RecordAttribute{arg[d_year].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return arg["__broadcastTarget"];
              },
              DegreeOfParallelism{dop}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return eq(arg[d_weeknuminyear], 6) & eq(arg[d_year], 1994);
          });

  auto rel =
      factory.getBuilder()
          .scan(lineorder,
                {lo_orderdate, lo_quantity, lo_discount, lo_extendedprice},
                catalog, pg{"distributed-block"})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {
                    new RecordAttribute{arg[lo_orderdate].getRegisteredAs()},
                    new RecordAttribute{arg[lo_quantity].getRegisteredAs()},
                    new RecordAttribute{arg[lo_discount].getRegisteredAs()},
                    new RecordAttribute{
                        arg[lo_extendedprice].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> {
                return (int)InfiniBandManager::server_id();  // std::nullopt;
              },
              DegreeOfParallelism{2}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .memmove_scaleout(16 * 1024)
          .router(DegreeOfParallelism{dop}, slack, RoutingPolicy::LOCAL,
                  DeviceType::CPU)
          .unpack()
          .filter([&](const auto &arg) -> expression_t {
            return ge(arg[lo_discount], 5) & le(arg[lo_discount], 7) &
                   ge(arg[lo_quantity], 26) & le(arg[lo_quantity], 35);
          })
          .join(
              rel_date,
              [&](const auto &build_arg) -> expression_t {
                return build_arg[d_datekey];
              },
              [&](const auto &build_arg) -> std::vector<GpuMatExpr> {
                return {};
              },
              {64},
              [&](const auto &probe_arg) -> expression_t {
                return probe_arg[lo_orderdate];
              },
              [&](const auto &probe_arg) -> std::vector<GpuMatExpr> {
                return {GpuMatExpr{
                    (probe_arg[lo_discount] * probe_arg[lo_extendedprice])
                        .as(lineorder, revenue),
                    1, 0}};
              },
              {64, 32}, 4, 16)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .router(DegreeOfParallelism{1}, slack, RoutingPolicy::RANDOM,
                  DeviceType::CPU)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .router_scaleout(
              [&](const auto &arg) -> std::vector<RecordAttribute *> {
                return {new RecordAttribute{arg[revenue].getRegisteredAs()}};
              },
              [&](const auto &arg) -> std::optional<expression_t> { return 0; },
              DegreeOfParallelism{1}, slack, RoutingPolicy::HASH_BASED,
              DeviceType::CPU, 1)
          .reduce(
              [&](const auto &arg) -> std::vector<expression_t> {
                return {arg[revenue]};
              },
              {SUM})
          .print(pg{"pm-csv"});

  return rel.prepare();
}

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

  // auto rel_date =
  //     RelBuilder{ctx}
  //         .scan<BinaryBlockPlugin>(date, {d_datekey}, catalog)
  //         .membrdcst_scaleout(2, false)
  //         .router_scaleout(
  //             [&](const auto &arg) -> std::vector<RecordAttribute *> {
  //               return {new
  //               RecordAttribute{arg[d_datekey].getRegisteredAs()}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> {
  //               return arg["__broadcastTarget"];
  //             },
  //             2, 1, slack, RoutingPolicy::HASH_BASED, DeviceType::CPU)
  //         .unpack();

  // auto rel =
  //     RelBuilder{ctx}
  //         .scan<BinaryBlockPlugin>(lineorder, {lo_orderdate}, catalog)
  //         .router_scaleout(
  //             [&](const auto &arg) -> std::vector<RecordAttribute *> {
  //               return {
  //                   new
  //                   RecordAttribute{arg[lo_orderdate].getRegisteredAs()}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> {
  //               return std::nullopt;
  //             },
  //             2, 1, slack, RoutingPolicy::RANDOM, DeviceType::CPU)
  //         .memmove_scaleout(slack)
  //         .unpack()
  //         .join(
  //             rel_date,
  //             [&](const auto &build_arg) -> expression_t {
  //               return build_arg[d_datekey];
  //             },
  //             [&](const auto &build_arg) -> std::vector<GpuMatExpr> {
  //               return {};
  //             },
  //             {64},
  //             [&](const auto &probe_arg) -> expression_t {
  //               return probe_arg[lo_orderdate];
  //             },
  //             [&](const auto &probe_arg) -> std::vector<GpuMatExpr> {
  //               return {GpuMatExpr{probe_arg[lo_orderdate], 1, 0}};
  //             },
  //             {64, 32}, 9, 1024)
  //         .reduce(
  //             [&](const auto &arg) -> std::vector<expression_t> {
  //               return {arg[lo_orderdate]};
  //             },
  //             {SUM})
  //         .router_scaleout(
  //             [&](const auto &arg) -> std::vector<RecordAttribute *> {
  //               return {
  //                   new
  //                   RecordAttribute{arg[lo_orderdate].getRegisteredAs()}};
  //             },
  //             [&](const auto &arg) -> std::optional<expression_t> { return
  //             0;
  //             }, 1, 1, slack, RoutingPolicy::HASH_BASED, DeviceType::CPU)
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

  // auto rel = ssb_q11_scaleout(ctx, catalog, slack);

  // auto statement = rel.prepare();

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

  // auto statement = ssb_q11(ctx, catalog, slack);
  // auto statement = ssb_q11_par(ctx, catalog, 16);
  // auto statement = ssb_q13_par(ctx, catalog, 16);
  // auto statement = ssb_q11_scaleout(ctx, catalog, slack);
  auto statement = ssb_q11_par_scaleout(ctx2, catalog2, 16);
  // auto statement = scan_sum_scaleout(ctx, catalog, 8);
  // auto statement = ssb_q13_par_scaleout(ctx, catalog, 16);

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
