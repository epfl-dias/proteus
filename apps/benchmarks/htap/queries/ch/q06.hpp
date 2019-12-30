/*
    Harmonia -- High-performance elastic HTAP on heterogeneous hardware.

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

#ifndef HARMONIA_QUERIES_CH_Q6_HPP_
#define HARMONIA_QUERIES_CH_Q6_HPP_

#include "../queries.hpp"

template <typename Tplugin>
PreparedStatement q_ch6_c1t() {
  std::string revenue = "revenue";
  RelBuilderFactory ctx{__FUNCTION__};
  CatalogParser &catalog = CatalogParser::getInstance();
  return ctx.getBuilder()
      .scan<Tplugin>(tpcc_orderline, {ol_delivery_d, ol_quantity, ol_amount},
                     catalog)
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return ge(arg[ol_quantity], 1) & le(arg[ol_quantity], 100000) &
               ge(arg[ol_delivery_d], expressions::DateConstant(915148800000)) &
               lt(arg[ol_delivery_d], expressions::DateConstant(1577836800000));
      })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_amount]};
          },
          {SUM})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_amount].as(outrel, revenue)};
      })
      .prepare();
}

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<6>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                             Tr aff_reduce, DeviceType dev) {
  RelBuilderFactory ctx{"ch_Q" + std::to_string(Qid) + "_" +
                        typeid(Tplugin).name()};
  CatalogParser &catalog = CatalogParser::getInstance();
  std::string revenue = "revenue";
  auto rel =
      ctx.getBuilder()
          .scan<Tplugin>(tpcc_orderline,
                         {ol_delivery_d, ol_quantity, ol_amount}, catalog)
          .router(dop, 1, RoutingPolicy::LOCAL, dev, aff_parallel());

  if (dev == DeviceType::GPU) {
    rel = rel.memmove(8, dev == DeviceType::CPU).to_gpu();
  }

  rel = rel.unpack()
            .filter([&](const auto &arg) -> expression_t {
              return ge(arg[ol_quantity], 1) & le(arg[ol_quantity], 100000) &
                     ge(arg[ol_delivery_d],
                        expressions::DateConstant(915148800000)) &
                     lt(arg[ol_delivery_d],
                        expressions::DateConstant(1577836800000));
            })
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_amount]};
                },
                {SUM});

  if (dev == DeviceType::GPU) {
    rel = rel.to_cpu();
  }

  rel = rel.router(DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM,
                   DeviceType::CPU, aff_reduce())
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_amount]};
                },
                {SUM})
            .print([&](const auto &arg,
                       std::string outrel) -> std::vector<expression_t> {
              return {arg[ol_amount].as(outrel, revenue)};
            });

  return rel.prepare();
}
template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<-6>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                              Tr aff_reduce, DeviceType dev) {
  assert(dev == DeviceType::CPU);
  RelBuilderFactory ctx{"ch_Q" + std::to_string(Qid) + "_" +
                        typeid(Tplugin).name()};
  CatalogParser &catalog = CatalogParser::getInstance();
  std::string revenue = "revenue";
  return ctx.getBuilder()
      .scan<Tplugin>(tpcc_orderline, {ol_delivery_d, ol_quantity, ol_amount},
                     catalog)
      .router(dop, 1, RoutingPolicy::RANDOM, DeviceType::CPU, aff_parallel())
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return ge(arg[ol_quantity], 1) & le(arg[ol_quantity], 100000) &
               ge(arg[ol_delivery_d], expressions::DateConstant(915148800000)) &
               lt(arg[ol_delivery_d], expressions::DateConstant(1577836800000));
      })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_amount]};
          },
          {SUM})
      .router(DegreeOfParallelism{1}, 1024, RoutingPolicy::RANDOM,
              DeviceType::CPU, aff_reduce())
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_amount]};
          },
          {SUM})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_amount].as(outrel, revenue)};
      })
      .prepare();
}
#endif /* HARMONIA_QUERIES_CH_Q6_HPP_ */
