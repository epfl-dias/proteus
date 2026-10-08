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

#include <cstdint>
#include <tpch/query.hpp>
#include <vector>

// TPC-H Q19 -- Discounted Revenue.
//
//   select sum(l_extendedprice * (1 - l_discount)) as revenue
//   from lineitem, part
//   where p_partkey = l_partkey
//     and (
//       (p_brand='Brand#12' and p_container in ('SM CASE','SM BOX','SM PACK',
//         'SM PKG') and l_quantity between 1 and 11 and p_size between 1 and 5
//         and l_shipmode in ('AIR','AIR REG') and l_shipinstruct='DELIVER IN
//         PERSON')
//       or (p_brand='Brand#23' and p_container in ('MED BAG','MED BOX','MED PKG',
//         'MED PACK') and l_quantity between 10 and 20 and p_size between 1 and
//         10 and l_shipmode in ('AIR','AIR REG') and l_shipinstruct='DELIVER IN
//         PERSON')
//       or (p_brand='Brand#34' and p_container in ('LG CASE','LG BOX','LG PACK',
//         'LG PKG') and l_quantity between 20 and 30 and p_size between 1 and 15
//         and l_shipmode in ('AIR','AIR REG') and l_shipinstruct='DELIVER IN
//         PERSON'));
//
// The query is a single big disjunctive predicate over lineitem ⋈ part with one
// scalar SUM -- no group-by, no order-by, no subquery (one of the genuinely flat
// non-nested TPC-H queries). Strings (p_brand, p_container, l_shipmode,
// l_shipinstruct) are dstring: eq(col, "literal") resolves the literal to its
// dictionary code, and `IN (...)` is expressed as an OR of dictionary equalities
// -- so no new operator is needed.
//
// SIMPLIFICATIONS (value-level only; the query *shape* -- the join and the
// 3-branch disjunction -- is preserved): money is scaled int, so
// l_extendedprice*(1-l_discount) becomes l_extendedprice*(100-l_discount)
// (kept in the integer/Int64 domain, exactly as Q14 does); p_size BETWEEN 1 and
// X is le(p_size, X) since TPC-H sizes are >= 1.

constexpr auto query = "tpch100_Q19_simplified";

PreparedStatement tpch::Query::prepare19(proteus::QueryShaper& morph) {
  morph.setQueryName(query);

  auto rel_part =
      morph.scan("part", {"p_partkey", "p_brand", "p_container", "p_size"});
  auto rel_lineitem = morph.scan(
      "lineitem", {"l_partkey", "l_quantity", "l_extendedprice", "l_discount",
                   "l_shipmode", "l_shipinstruct"});

  return morph
      .parallel(
          rel_lineitem, {rel_part},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto part_d = build.at(0).unpack().project(
                [&](const auto& arg) -> std::vector<expression_t> {
                  return {arg["p_partkey"], arg["p_brand"], arg["p_container"],
                          arg["p_size"]};
                });

            return probe.unpack()
                // shipmode/shipinstruct are common to all 3 branches -> push
                // them to the probe (lineitem) before the join to shrink it.
                .filter([&](const auto& arg) -> expression_t {
                  // dbgen's mode value is "REG AIR" (the canonical Q19 text
                  // 'AIR REG' is a spec typo); literals must match the dict.
                  auto shipmode = eq(arg["l_shipmode"], "AIR") |
                                  eq(arg["l_shipmode"], "REG AIR");
                  auto shipinstruct =
                      eq(arg["l_shipinstruct"], "DELIVER IN PERSON");
                  return shipmode & shipinstruct;
                })
                .join(
                    part_d,
                    [&](const auto& build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto& probe_arg) -> expression_t {
                      return probe_arg["l_partkey"];
                    })
                // the 3-branch disjunction over (p_brand, p_container, p_size,
                // l_quantity); IN(...) -> OR of dstring equalities.
                .filter([&](const auto& arg) -> expression_t {
                  auto b1 =
                      eq(arg["p_brand"], "\"Brand#12\"") &
                      (eq(arg["p_container"], "SM CASE") |
                       eq(arg["p_container"], "SM BOX") |
                       eq(arg["p_container"], "SM PACK") |
                       eq(arg["p_container"], "SM PKG")) &
                      ge(arg["l_quantity"], 1) & le(arg["l_quantity"], 11) &
                      le(arg["p_size"], 5);
                  auto b2 =
                      eq(arg["p_brand"], "\"Brand#23\"") &
                      (eq(arg["p_container"], "MED BAG") |
                       eq(arg["p_container"], "MED BOX") |
                       eq(arg["p_container"], "MED PKG") |
                       eq(arg["p_container"], "MED PACK")) &
                      ge(arg["l_quantity"], 10) & le(arg["l_quantity"], 20) &
                      le(arg["p_size"], 10);
                  auto b3 =
                      eq(arg["p_brand"], "\"Brand#34\"") &
                      (eq(arg["p_container"], "LG CASE") |
                       eq(arg["p_container"], "LG BOX") |
                       eq(arg["p_container"], "LG PACK") |
                       eq(arg["p_container"], "LG PKG")) &
                      ge(arg["l_quantity"], 20) & le(arg["l_quantity"], 30) &
                      le(arg["p_size"], 15);
                  return b1 | b2 | b3;
                })
                .reduce(
                    [&](const auto& arg) -> std::vector<expression_t> {
                      const expression_t discount = 100 - arg["l_discount"];
                      const auto revenue =
                          (arg["l_extendedprice"].template as<Int64Type>() *
                           expressions::CastExpression(new Int64Type(),
                                                       discount));
                      return {expression_t{revenue}.as("tmp", "revenue")};
                    },
                    {SUM});
          })
      .reduce(
          [&](const auto& arg) -> std::vector<expression_t> {
            return {arg["revenue"]};
          },
          {SUM})
      .print(pg{"pm-csv"})
      .prepare();
}
