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

#ifndef HARMONIA_QUERIES_CH_Q1_HPP_
#define HARMONIA_QUERIES_CH_Q1_HPP_

#include "../queries.hpp"

template <typename Tplugin>
PreparedStatement q_ch_c1t() {
  std::string count_order = "count_order";
  RelBuilderFactory ctx{__FUNCTION__};
  CatalogParser &catalog = CatalogParser::getInstance();
  return ctx.getBuilder()
      .scan<Tplugin>(tpcc_orderline,
                     {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                     catalog)
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg[ol_delivery_d],
                  expressions::DateConstant(/*FIX*/ 904694400000));
      })
      .reduce(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                    expression_t{1}.as(tpcc_orderline, count_order)};
          },
          {SUM /*fix*/, SUM, SUM, SUM})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        std::vector<expression_t> ret{
            arg[ol_number].as(outrel, ol_number),
            arg[ol_quantity].as(outrel, "sum_qty"),
            arg[ol_amount].as(outrel, "sum_amount"),
            (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
            (arg[ol_amount] / arg[count_order].template as<FloatType>())
                .as(outrel, "avg_amount"),
            arg[count_order].as(outrel, count_order)};
        return ret;
      })
      .prepare();
}

template <typename Tplugin>
PreparedStatement q_ch_cpar(DegreeOfParallelism dop,
                            std::unique_ptr<Affinitizer> aff_parallel,
                            std::unique_ptr<Affinitizer> aff_reduce,
                            DeviceType dev = DeviceType::CPU) {
  std::string count_order = "count_order";
  RelBuilderFactory ctx{__FUNCTION__};
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel =
      ctx.getBuilder()
          .scan<Tplugin>(tpcc_orderline,
                         {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                         catalog)
          .router(dop, 1, RoutingPolicy::RANDOM, dev, std::move(aff_parallel));

  if (dev == DeviceType::GPU) rel = rel.to_gpu();

  rel = rel.unpack()
            .filter([&](const auto &arg) -> expression_t {
              return gt(arg[ol_delivery_d],
                        expressions::DateConstant(/*FIX*/ 904694400000));
            })
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                          expression_t{1}.as(tpcc_orderline, count_order)};
                },
                {SUM /*fix*/, SUM, SUM, SUM});

  if (dev == DeviceType::GPU) rel = rel.to_cpu();

  rel = rel.router(DegreeOfParallelism{1}, 1, RoutingPolicy::RANDOM,
                   DeviceType::CPU, std::move(aff_reduce))
            .reduce(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                          arg[count_order]};
                },
                {SUM /*fix*/, SUM, SUM, SUM})
            .print([&](const auto &arg,
                       std::string outrel) -> std::vector<expression_t> {
              std::vector<expression_t> ret{
                  arg[ol_number].as(outrel, ol_number),
                  arg[ol_quantity].as(outrel, "sum_qty"),
                  arg[ol_amount].as(outrel, "sum_amount"),
                  (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
                  (arg[ol_amount] / arg[count_order].template as<FloatType>())
                      .as(outrel, "avg_amount"),
                  arg[count_order].as(outrel, count_order)};
              return ret;
            });

  return rel.prepare();
}

// PreparedStatement q_ch2_c1t() {
//   return PreparedStatement::from(
//       "/scratch/chrysoge/pelago_sigmod2020_htap/src/htap/ch-plans/q1.json",
//       __FUNCTION__);
// }

template <typename Tplugin>
PreparedStatement q_ch(DegreeOfParallelism dop,
                       std::unique_ptr<Affinitizer> aff_parallel,
                       std::unique_ptr<Affinitizer> aff_reduce) {
  if (dop == DegreeOfParallelism{1}) return q_ch_c1t<Tplugin>();
  return q_ch_cpar<Tplugin>(dop, std::move(aff_parallel),
                            std::move(aff_reduce));
}

template <typename Tplugin>
PreparedStatement q_ch1_c1t() {
  std::string count_order = "count_order";
  RelBuilderFactory ctx{__FUNCTION__};
  CatalogParser &catalog = CatalogParser::getInstance();
  return ctx.getBuilder()
      .scan<Tplugin>(tpcc_orderline,
                     {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                     catalog)
      .unpack()
      .filter([&](const auto &arg) -> expression_t {
        return gt(arg[ol_delivery_d],
                  expressions::DateConstant(/*FIX*/ 904694400000));
      })
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {
                GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                GpuAggrMatExpr{arg[ol_amount], 2, 0, SUM},
                GpuAggrMatExpr{expression_t{1}.as(tpcc_orderline, count_order),
                               3, 0, SUM}};
          },
          10, 1024 * 1024)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                    arg[count_order]};
          },
          {direction::ASC, direction::NONE, direction::NONE})
      .print([&](const auto &arg,
                 std::string outrel) -> std::vector<expression_t> {
        return {arg[ol_number].as(outrel, ol_number),
                arg[ol_quantity].as(outrel, "sum_qty"),
                arg[ol_amount].as(outrel, "sum_amount"),
                (arg[ol_quantity] / arg[count_order]).as(outrel, "avg_qty"),
                (arg[ol_amount] / arg[count_order].template as<FloatType>())
                    .as(outrel, "avg_amount"),
                arg[count_order].as(outrel, count_order)};
      })
      .prepare();
}

template <>
template <typename Tplugin, typename Tp, typename Tr>
PreparedStatement Q<1>::cpar(DegreeOfParallelism dop, Tp aff_parallel,
                             Tr aff_reduce, DeviceType dev) {
  std::string count_order = "count_order";
  RelBuilderFactory ctx{"ch_Q" + std::to_string(Qid) + "_" +
                        typeid(Tplugin).name()};
  CatalogParser &catalog = CatalogParser::getInstance();
  auto rel =
      ctx.getBuilder()
          .scan<Tplugin>(tpcc_orderline,
                         {ol_delivery_d, ol_number, ol_amount, ol_quantity},
                         catalog)
          .router(dop, 8, RoutingPolicy::RANDOM, dev, aff_parallel())
      //      .memmove(8, dev == DeviceType::CPU)
      ;

  if (dev == DeviceType::GPU) {
    rel = rel.memmove(8, dev == DeviceType::CPU).to_gpu();
  }

  rel = rel.unpack()
            // .filter([&](const auto &arg) -> expression_t {
            //   return gt(arg[ol_delivery_d],
            //             expressions::DateConstant(/*FIX*/ 904694400000));
            // })
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                          GpuAggrMatExpr{
                              expression_t{1}.as(tpcc_orderline, count_order),
                              1, 32, SUM},
                          GpuAggrMatExpr{arg[ol_amount], 1, 64, SUM}};
                },
                5, 128 * 1024);

  if (dev == DeviceType::GPU) {
    rel = rel.to_cpu();
  }

  rel = rel.router(DegreeOfParallelism{1}, 128, RoutingPolicy::RANDOM,
                   DeviceType::CPU, aff_reduce())
            .groupby(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number]};
                },
                [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                  return {GpuAggrMatExpr{arg[ol_quantity], 1, 0, SUM},
                          GpuAggrMatExpr{arg[ol_amount], 2, 0, SUM},
                          GpuAggrMatExpr{arg[count_order], 3, 0, SUM}};
                },
                5, 128)
            .sort(
                [&](const auto &arg) -> std::vector<expression_t> {
                  return {arg[ol_number], arg[ol_quantity], arg[ol_amount],
                          arg[count_order]};
                },
                {direction::ASC, direction::NONE, direction::NONE})
            .print([&](const auto &arg,
                       std::string outrel) -> std::vector<expression_t> {
              return {arg[ol_number].as(outrel, ol_number),
                      arg[ol_quantity].as(outrel, "sum_qty"),
                      arg[ol_amount].as(outrel, "sum_amount"),
                      (arg[ol_quantity] / (arg[count_order] + 1))
                          .as(outrel, "avg_qty"),
                      (arg[ol_amount] /
                       (arg[count_order] + 1).template as<FloatType>())
                          .as(outrel, "avg_amount"),
                      arg[count_order].as(outrel, count_order)};
            });

  return rel.prepare();
}

#endif /* HARMONIA_QUERIES_CH_Q1_HPP_ */
