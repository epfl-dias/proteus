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
#include <fstream>
#include <glog/logging.h>
#include <string>
#include <tpch/query.hpp>
#include <vector>

// TPC-H Q14 -- Promotion Effect (simplified, integer/dict).
//
//   100 * SUM(CASE WHEN p_type LIKE 'PROMO%' THEN discounted_revenue ELSE 0 END)
//       / SUM(discounted_revenue)
//   from lineitem, part
//   where l_partkey = p_partkey
//     and l_shipdate >= date '1995-09-01'
//     and l_shipdate <  date '1995-09-01' + interval '1' month;
//
// The only non-trivial piece is `p_type LIKE 'PROMO%'`. p_type is a dstring
// (int32 dictionary code). Because the dictionary is a bijection value<->code,
// the pattern match is *exactly* set membership over the codes whose value
// matches the pattern:
//
//   p_type LIKE 'PROMO%'  ==  p_type IN { dict entries starting with 'PROMO' }
//
// So we do the "dictionary pre-match" on the host at plan time -- read the
// p_type dictionary, evaluate the constant pattern once per entry -- and lower
// the result to an OR of dstring equalities (exactly the dict-code path Q12/Q19
// already use). This is semantically exact, needs no new device code and does
// not touch the decode core (buffer-manager.cu); correctness is demonstrated by
// the byte-identity gate vs the UNCOM answer.
//
// Value-level simplifications (query *shape* preserved): money is scaled int, so
// l_extendedprice*(1-l_discount) becomes l_extendedprice*(100-l_discount) kept in
// the Int64 domain (exactly as Q19). The final float ratio is left to the
// consumer; the benchmark prints both integer sums.

namespace {

// Read the p_type dictionary (<value>:<code> per line) and return the values
// matching LIKE 'PROMO%'. 'PROMO%' is a constant prefix pattern, so the host
// match is a plain prefix test -- no runtime LIKE evaluator needed.
std::vector<std::string> promoTypesMatching() {
  const std::string dict_path = "inputs/tpch100/part.csv.p_type.dict";
  const std::string prefix = "PROMO";  // LIKE 'PROMO%'
  std::ifstream dict(dict_path);
  CHECK(dict.good()) << "cannot open p_type dictionary: " << dict_path;
  std::vector<std::string> matches;
  std::string line;
  while (std::getline(dict, line)) {
    const auto sep = line.find_last_of(':');
    CHECK_NE(sep, std::string::npos) << "invalid dictionary line: " << line;
    const std::string value = line.substr(0, sep);
    if (value.rfind(prefix, 0) == 0) matches.push_back(value);
  }
  CHECK(!matches.empty()) << "no p_type matched LIKE 'PROMO%' in " << dict_path;
  return matches;
}

}  // namespace

constexpr auto query = "tpch100_Q14_simplified";

PreparedStatement tpch::Query::prepare14(proteus::QueryShaper& morph) {
  morph.setQueryName(query);

  auto rel_part = morph.scan("part", {"p_partkey", "p_type"});
  auto rel_lineitem = morph.scan(
      "lineitem", {"l_partkey", "l_shipdate", "l_extendedprice", "l_discount"});

  const std::vector<std::string> promo = promoTypesMatching();

  return morph
      .parallel(
          rel_lineitem, {rel_part},
          [&](RelBuilder probe, std::vector<RelBuilder> build) {
            auto part_d = build.at(0).unpack().project(
                [&](const auto& arg) -> std::vector<expression_t> {
                  return {arg["p_partkey"], arg["p_type"]};
                });

            return probe.unpack()
                .filter([&](const auto& arg) -> expression_t {
                  return expressions::hint(
                      ge(arg["l_shipdate"], 19950901) &
                          lt(arg["l_shipdate"], 19951001),
                      expressions::Selectivity(1.0 / (7.0 * 12.0)));
                })
                .join(
                    part_d,
                    [&](const auto& build_arg) -> expression_t {
                      return build_arg["p_partkey"];
                    },
                    [&](const auto& probe_arg) -> expression_t {
                      return probe_arg["l_partkey"];
                    })
                .reduce(
                    [&](const auto& arg) -> std::vector<expression_t> {
                      const expression_t discount = 100 - arg["l_discount"];
                      const auto revenue =
                          (arg["l_extendedprice"].template as<Int64Type>() *
                           expressions::CastExpression(new Int64Type(),
                                                       discount));
                      // p_type LIKE 'PROMO%' == OR of dstring equalities over
                      // the pre-matched dictionary entries (host pre-match).
                      expression_t is_promo = eq(arg["p_type"], promo[0]);
                      for (size_t i = 1; i < promo.size(); ++i) {
                        is_promo = is_promo | eq(arg["p_type"], promo[i]);
                      }
                      return {cond(is_promo, revenue, expression_t{int64_t{0}})
                                  .as("tmp", "promo_revenue"),
                              expression_t{revenue}.as("tmp", "total_revenue")};
                    },
                    {SUM, SUM});
          })
      .reduce(
          [&](const auto& arg) -> std::vector<expression_t> {
            return {arg["promo_revenue"], arg["total_revenue"]};
          },
          {SUM, SUM})
      .print(pg{"pm-csv"})
      .prepare();
}
