/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2026
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

#ifndef PROTEUS_TPCH100_QUERY_HPP
#define PROTEUS_TPCH100_QUERY_HPP

#include <olap/plan/prepared-statement.hpp>
#include <query-shaping/input-prefix-query-shaper.hpp>
#include <query-shaping/query-shaper.hpp>

// TPC-H prepared statements used by the CB2T evaluation, built with the same
// RelBuilder/QueryShaper infrastructure as the SSB benchmark (see ../ssb).
//
// DATA-MODEL ASSUMPTION (matches the SSB ssbm100 catalog convention):
//   * dates  are stored as int in YYYYMMDD form (e.g. 1998-09-02 -> 19980902)
//   * money / rates (l_extendedprice, l_discount, l_tax, ...) are stored as
//     scaled int (cents / integer percentage points), exactly like SSB's
//     lo_extendedprice / lo_discount / lo_revenue.
//   * char columns (l_returnflag, l_linestatus, c_mktsegment, ...) are dstring.
// Aggregate arithmetic is therefore kept in the integer domain and the heavy
// TPC-H money formulas are simplified to single products (as SSB Q1 already
// does: revenue = extendedprice * discount). The query *shape* -- every join,
// filter predicate and group-by -- is preserved; only the value expressions
// are simplified. The measured GPU work (scan + decode + filter + join +
// group-by over the listed columns) is unchanged.

namespace tpch {
class Query {
 public:
  static PreparedStatement prepare05(proteus::QueryShaper &morph);  // Q5
  static PreparedStatement prepare12(proteus::QueryShaper &morph);  // Q12

  static std::vector<PreparedStatement> prepareAll(
      proteus::QueryShaper &shaper) {
    return {prepare05(shaper), prepare12(shaper)};
  }

  static std::map<std::string,
                  std::function<double(proteus::InputPrefixQueryShaper &)>>
  getStats(size_t SF);
};

}  // namespace tpch

template <typename T>
std::unique_ptr<T> make_shaper(
    size_t SF, decltype(tpch::Query::getStats(std::declval<size_t>())) stat) {
  return std::make_unique<T>("inputs/tpch" + std::to_string(SF) + "/",
                             std::move(stat));
}

#endif /* PROTEUS_TPCH100_QUERY_HPP */
