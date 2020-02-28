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

#ifndef HARMONIA_QUERIES_CH_QSTOCK_HPP_
#define HARMONIA_QUERIES_CH_QSTOCK_HPP_

#include <operators/relbuilder-factory.hpp>

#include "ch-queries.hpp"
#include "queries/query-interface.hpp"

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<-1>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                              Tr aff_reduce, DeviceType dev) {
  assert(dev == DeviceType::CPU);
  RelBuilderFactory ctx{"ch_Q" + std::to_string(Qid) + "_" +
                        typeid(Tplugin).name()};
  CatalogParser &catalog = CatalogParser::getInstance();
  return ctx.getBuilder()
      .scan<Tplugin>("tpcc_stock", {"s_i_id", "s_w_id", "s_quantity"}, catalog)
      // (table=[[SSB, tpcc_stock]], fields=[[0, 1, 2]],
      // traits=[Pelago.[].packed.X86_64.homSingle.hetSingle.none])
      .router(dop, 1, RoutingPolicy::LOCAL, DeviceType::CPU, aff_parallel())
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.none])
      .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .project([&](const auto &arg) -> std::vector<expression_t> {
        return {(arg["$1"]).as("PelagoProject#1853", "$0"),
                (arg["$0"]).as("PelagoProject#1853", "$1"),
                (arg["$2"]).as("PelagoProject#1853", "$2")};
      })
      // (s_w_id=[$1], s_i_id=[$0], s_quantity=[$2],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#1854", "$0"),
                    arg["$1"].as("PelagoAggregate#1854", "$1")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#1854", "$2"),
                                   1, 0, SUM}};
          },
          10, 128 * 1024 * 1024)
      // (group=[{0, 1}], EXPR$2=[SUM($2)],
      // trait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64])
      // .pack()
      // (trait=[Pelago.[].packed.X86_64.homRandom.hetSingle.cX86_64],
      // intrait=[Pelago.[].unpckd.X86_64.homRandom.hetSingle.cX86_64],
      // inputRows=[5.12E9], cost=[{5865.4596 rows, 5859.6 cpu, 0.0 io}])
      .router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
              DeviceType::CPU, aff_reduce())
      // (trait=[Pelago.[].packed.X86_64.homSingle.hetSingle.cX86_64])
      // .unpack()
      // (trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["$0"].as("PelagoAggregate#1858", "$0"),
                    arg["$1"].as("PelagoAggregate#1858", "$1")};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{(arg["$2"]).as("PelagoAggregate#1858", "$2"),
                                   1, 0, SUM}};
          },
          24, 128 * 1024 * 1024)
      // (group=[{0, 1}], EXPR$2=[SUM($2)],
      // trait=[Pelago.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg["$0"].as(outrel, "s_w_id"), arg["$1"].as(outrel, "s_i_id"),
                arg["$2"].as(outrel, "EXPR$2")};
      })
      // (trait=[ENUMERABLE.[].unpckd.X86_64.homSingle.hetSingle.cX86_64])
      .prepare();
}

#endif /* HARMONIA_QUERIES_CH_QSTOCK_HPP_ */
