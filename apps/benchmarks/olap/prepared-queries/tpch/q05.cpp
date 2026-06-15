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

// TPC-H Q5 -- Local Supplier Volume.
//   select n_name, sum(l_extendedprice*(1-l_discount)) as revenue
//   from   customer, orders, lineitem, supplier, nation, region
//   where  c_custkey   = o_custkey
//     and  l_orderkey  = o_orderkey
//     and  l_suppkey   = s_suppkey
//     and  c_nationkey = s_nationkey            -- CYCLE: customer & supplier
//     and  s_nationkey = n_nationkey            --        must share a nation
//     and  n_regionkey = r_regionkey
//     and  r_name      = 'ASIA'
//     and  o_orderdate >= date '1994-01-01'
//     and  o_orderdate <  date '1994-01-01' + interval '1' year
//   group by n_name
//   order by revenue desc;
//
// Join-bound query, and the richest join in our set: a SIX-way join whose graph
// is NOT a tree. The predicate c_nationkey = s_nationkey couples the customer
// and supplier branches (each lineitem's supplier and its order's customer must
// reside in the same nation), so the join graph contains a cycle -- a regime
// neither SSB's star joins (Q4.1) nor TPC-H's two-large-table join (Q3) reach.
//
// Dataflow (left-deep, probe = lineitem fact table):
//   lineitem
//     --join(o_orderkey = l_orderkey)--> orders     (brings o_custkey)
//     --join(c_custkey  = o_custkey )--> customer    (brings c_nationkey)
//     --join(s_suppkey  = l_suppkey )--> supplier    (brings s_nationkey)
//     --join(n_nationkey= s_nationkey)--> nation      (brings n_name, n_regionkey)
//     --join(r_regionkey= n_regionkey)--> region      (filtered r_name='ASIA')
//   The cyclic equality c_nationkey = s_nationkey is enforced as a RESIDUAL
//   filter once both nationkeys are in the probe stream (Proteus's .join() is a
//   single-key equi-join, so a non-tree predicate cannot be a join key here).
//   Joining the ASIA-filtered region acts as a semi-join restricting nation to
//   the ASIA region.
//
// SIMPLIFICATIONS vs. canonical TPC-H Q5 (every predicate and all five joins are
// preserved; the join graph -- including the cycle -- is intact):
//   * revenue formula reduced to a single integer product, as in SSB / our Q3:
//        sum(l_extendedprice*(1-l_discount)) -> sum(l_extendedprice*l_discount)
//     keeps l_discount in the scan; result fits int32.
//   * Encoding: o_orderdate is int YYYYMMDD, so '1994-01-01' / '1995-01-01'
//     become 19940101 / 19950101; r_name and n_name are dstring; money is
//     scaled int. The string constant 'ASIA' is resolved to its dictionary
//     code at plan time (eq on a dstring is an int comparison on the GPU).
//
// NOTE (dataset): Q5 references columns beyond the original Q1/Q3/Q6 set:
//   lineitem.l_suppkey; customer.c_nationkey; supplier.s_suppkey,
//   supplier.s_nationkey; nation.n_nationkey, n_name, n_regionkey;
//   region.r_regionkey, r_name. These are included in adm-partition-tpch's
//   column set; regenerate the dataset before running Q5 on older layouts.
//
// NOTE (group cardinality): n_name has only 25 distinct values (<=5 after the
// ASIA region filter), so the group-by is tiny -- the cost is in the joins, not
// the aggregation. group_bits / group_max below are sized accordingly.

constexpr auto query = "tpch100_Q5";

PreparedStatement tpch::Query::prepare05(proteus::QueryShaper &morph) {
  morph.setQueryName(query);

  auto rel_customer = morph.scan("customer", {"c_custkey", "c_nationkey"});

  auto rel_orders =
      morph.scan("orders", {"o_orderkey", "o_custkey", "o_orderdate"});

  auto rel_supplier = morph.scan("supplier", {"s_suppkey", "s_nationkey"});

  auto rel_nation =
      morph.scan("nation", {"n_nationkey", "n_name", "n_regionkey"});

  auto rel_region = morph.scan("region", {"r_regionkey", "r_name"});

  auto rel = morph.scan(
      "lineitem", {"l_orderkey", "l_suppkey", "l_extendedprice", "l_discount"});

  // n_name has <=25 distinct values (<=5 within ASIA); a tiny group-by.
  const size_t group_bits = 6;        // 2^6 = 64 buckets, ample for 25 nations
  const size_t group_max = 1u << 6;

  return morph
      .parallel(
          rel, {rel_customer, rel_orders, rel_supplier, rel_nation, rel_region},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto customer_d =
                build.at(0).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["c_custkey"], arg["c_nationkey"]};
                    });

            auto orders_d =
                build.at(1)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          ge(arg["o_orderdate"], 19940101) &
                              lt(arg["o_orderdate"], 19950101),
                          expressions::Selectivity(1.0 / 6.5));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["o_orderkey"], arg["o_custkey"]};
                    });

            auto supplier_d =
                build.at(2).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["s_suppkey"], arg["s_nationkey"]};
                    });

            auto nation_d =
                build.at(3).unpack().project(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["n_nationkey"], arg["n_name"],
                              arg["n_regionkey"]};
                    });

            auto region_d =
                build.at(4)
                    .unpack()
                    .filter([&](const auto &arg) -> expression_t {
                      return expressions::hint(
                          eq(arg["r_name"], "ASIA"),
                          expressions::Selectivity(1.0 / 5));
                    })
                    .project([&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["r_regionkey"]};
                    });

            return probe.unpack()
                .join(
                    orders_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["o_orderkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["l_orderkey"];
                    },
                    26, 25'000'000)
                .join(
                    customer_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["c_custkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["o_custkey"];
                    },
                    25, 16'000'000)
                .join(
                    supplier_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["s_suppkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["l_suppkey"];
                    },
                    21, 1'100'000)
                // CYCLE: customer and supplier must be in the same nation.
                // Enforced as a residual equi-predicate now that both
                // nationkeys are in the probe stream.
                .filter([&](const auto &arg) -> expression_t {
                  return expressions::hint(
                      eq(arg["c_nationkey"], arg["s_nationkey"]),
                      expressions::Selectivity(1.0 / 25));
                })
                .join(
                    nation_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["n_nationkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["s_nationkey"];
                    },
                    12, 2048)
                .join(
                    region_d,
                    [&](const auto &build_arg) -> expression_t {
                      return build_arg["r_regionkey"];
                    },
                    [&](const auto &probe_arg) -> expression_t {
                      return probe_arg["n_regionkey"];
                    },
                    12, 2048)
                .groupby(
                    [&](const auto &arg) -> std::vector<expression_t> {
                      return {arg["n_name"].as("tmp", "n_name")};
                    },
                    [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
                      return {GpuAggrMatExpr{
                          (arg["l_extendedprice"] * arg["l_discount"])
                              .as("tmp", "revenue"),
                          1, 0, SUM}};
                    },
                    group_bits, group_max)
                .pack();
          })
      .unpack()
      .groupby(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["n_name"]};
          },
          [&](const auto &arg) -> std::vector<GpuAggrMatExpr> {
            return {GpuAggrMatExpr{arg["revenue"], 1, 0, SUM}};
          },
          group_bits, group_max)
      .sort(
          [&](const auto &arg) -> std::vector<expression_t> {
            return {arg["revenue"], arg["n_name"]};
          },
          {direction::DESC, direction::NONE})
      .print(pg{"pm-csv"})
      .prepare();
}
