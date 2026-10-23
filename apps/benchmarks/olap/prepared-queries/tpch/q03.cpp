/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
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

#include <tpch/query.hpp>

// TPC-H Q3 -- Shipping Priority.
//   select l_orderkey,
//          sum(l_extendedprice*(1-l_discount)) as revenue,
//          o_orderdate, o_shippriority
//   from   customer, orders, lineitem
//   where  c_mktsegment = 'BUILDING'
//     and  c_custkey   = o_custkey
//     and  l_orderkey  = o_orderkey
//     and  o_orderdate < date '1995-03-15'
//     and  l_shipdate  > date '1995-03-15'
//   group by l_orderkey, o_orderdate, o_shippriority
//   order by revenue desc, o_orderdate
//   limit 10;
//
// Join-bound query. Unlike SSB's single-fact star schema, Q3 joins TWO large
// tables (lineitem probe, orders build) plus the customer dimension -- the
// main reason to add a TPC-H query: it exercises CB2T feeding a multi-join +
// group-by + sort pipeline that SSB structurally cannot produce.
//
// Dataflow: build hash tables on the (filtered) customer and orders relations,
// then stream the (filtered) lineitem fact table through:
//   lineitem  --join(o_orderkey = l_orderkey)-->  orders
//             --join(c_custkey  = o_custkey )-->  customer
// After the first join o_custkey is in the probe stream, so it is available as
// the probe key of the second join.
//
// SIMPLIFICATIONS vs. canonical TPC-H Q3 (every predicate and both joins are
// preserved):
//   * revenue formula reduced to a single integer product, as in SSB:
//        sum(l_extendedprice*(1-l_discount)) -> sum(l_extendedprice*l_discount)
//     This keeps l_discount in the scan; result fits int32.
//   * LIMIT 10 is dropped. Proteus has no top-k operator; LIMIT here is a
//     presentation-only truncation of the already-sorted output and does not
//     affect the measured GPU work. The full sorted result is printed instead.
//   * Encoding: o_orderdate / l_shipdate are int YYYYMMDD, so the bound
//     '1995-03-15' becomes 19950315; c_mktsegment is dstring; money is scaled
//     int.
//
// NOTE: the group-by key l_orderkey is high-cardinality (~ a few million
// qualifying orders at SF=100), unlike any SSB group-by. The hash_bits /
// maxInputSize below are sized for that and may need tuning to the measured
// qualifying cardinality on the target GPU.

constexpr auto query = "tpch100_Q3";

PreparedStatement tpch::Query::prepare03(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_customer = morph.scan("customer", {"c_custkey", "c_mktsegment"});

  auto rel_orders = morph.scan(
      "orders", {"o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"});

  auto rel = morph.scan(
      "lineitem", {"l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"});

  // group cardinality hint: ~ qualifying orders
  //   ~ (SF * 1.5M orders) * P[orderdate<date] * P[mktsegment=BUILDING]
  const size_t partial_group_bits = 22;       // 2^22 buckets
  const size_t partial_group_max = 1u << 22;  // partial groups stay below 3M
  const size_t final_group_bits = 21;         // 2^21 buckets
  const size_t final_group_max = 1u << 21;    // final groups stay near 1.1M

  return morph
      .parallel(
          rel, {rel_customer, rel_orders},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto customer_d =
                build.at(0)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["c_mktsegment"], "BUILDING"),
                          expressions::Selectivity(1.0 / 5));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["c_custkey"]};
                    });

            auto orders_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          lt(arg["o_orderdate"], 19950315),
                          expressions::Selectivity(0.48));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["o_orderkey"], arg["o_custkey"],
                              arg["o_orderdate"], arg["o_shippriority"]};
                    });

            return probe.unpack()
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(gt(arg["l_shipdate"], 19950315),
                                           expressions::Selectivity(0.55));
                })
                .join(
                    orders_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["o_orderkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["l_orderkey"];
                    })
                .join(
                    customer_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["c_custkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["o_custkey"];
                    })
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["l_orderkey"].as("tmp", "l_orderkey"),
                              arg["o_orderdate"].as("tmp", "o_orderdate"),
                              arg["o_shippriority"].as("tmp", "o_shippriority")};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      return {GpuAggrMatExpr{
                          (arg["l_extendedprice"] * arg["l_discount"])
                              .as("tmp", "revenue"),
                          1, 0, SUM}};
                    },
                    partial_group_bits, partial_group_max)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["l_orderkey"], arg["o_orderdate"],
                    arg["o_shippriority"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["revenue"], 1, 0, SUM}};
          },
          final_group_bits, final_group_max)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["revenue"], arg["o_orderdate"], arg["l_orderkey"],
                    arg["o_shippriority"]};
          },
          {direction::DESC, direction::ASC, direction::NONE, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
