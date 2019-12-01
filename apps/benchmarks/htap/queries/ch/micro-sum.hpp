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

#ifndef HARMONIA_QUERIES_MICRO_SUM_HPP_
#define HARMONIA_QUERIES_MICRO_SUM_HPP_

#include "../queries.hpp"

template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement q_sum_c1t() {
  auto ctx = new ParallelContext("main2", false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<Tplugin>(tpcc_orderline, {ol_o_id}, catalog)
      .unpack()
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            // return {expression_t{1}.as(tpcc_orderline,ol_o_id)};
            return {arg[ol_o_id]};
          },
          {SUM})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_o_id].as(outrel, ol_o_id)};
      })
      .prepare();
}

template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement q_sum_cpar(DegreeOfParallelism dop,
                             std::unique_ptr<Affinitizer> aff_parallel,
                             std::unique_ptr<Affinitizer> aff_reduce) {
  auto ctx = new ParallelContext("main3", false);
  CatalogParser &catalog = CatalogParser::getInstance();
  return RelBuilder{ctx}
      .scan<Tplugin>(tpcc_orderline, {ol_o_id}, catalog)
      .router(dop, 1, RoutingPolicy::RANDOM, DeviceType::CPU,
              std::move(aff_parallel))
      .unpack()
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            // return {expression_t{1}.as(tpcc_orderline,ol_o_id)};
            return {arg[ol_o_id]};
          },
          {SUM})
      .router(DegreeOfParallelism{1}, 16, RoutingPolicy::LOCAL, DeviceType::CPU,
              std::move(aff_reduce))
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            // return {expression_t{1}.as(tpcc_orderline,ol_o_id)};
            return {arg[ol_o_id]};
          },
          {SUM})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_o_id].as(outrel, ol_o_id)};
      })
      .prepare();
}

template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement q_sum(DegreeOfParallelism dop,
                        std::unique_ptr<Affinitizer> aff_parallel,
                        std::unique_ptr<Affinitizer> aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_sum_c1t<Tplugin>();
  return q_sum_cpar<Tplugin>(dop, std::move(aff_parallel),
                             std::move(aff_reduce));
}

#endif /* HARMONIA_QUERIES_MICRO_SUM_HPP_ */
