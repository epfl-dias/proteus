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

#ifndef PROTEUS_STACK_QUERY_HPP
#define PROTEUS_STACK_QUERY_HPP

#include <olap/plan/prepared-statement.hpp>
#include <query-shaping/input-prefix-query-shaper.hpp>
#include <query-shaping/query-shaper.hpp>

// Stack (StackOverflow, Bao SIGMOD'21) prepared statements, built with the
// same RelBuilder/QueryShaper infrastructure as the TPC-H/SSB benchmarks
// (see ../tpch). Six fixed query instances (SO-Q1/Q4/Q6/Q11/Q12/Q14) chosen
// for the VLDB'27 revision -- rationale, rewrites and verification are in
// Query set used by the CB2T evaluation.
//
// DATA-MODEL ASSUMPTION (export conventions, matching the SSB/TPC-H catalog):
//   * timestamps are int YYYYMMDD (creation_date); "+ 1 year" is + 10000.
//   * SQL NULLs in int columns were exported as the sentinel -1 (never equal
//     to a real id; fails every >=0-bounded range predicate).
//   * site_name / tag name / badge name are dstring dictionaries.
//   * ids are only unique per (site_id, id). Single-site queries push the
//     site_id literal into per-table filters and join on id alone;
//     multi-site queries join on a packed int64 key (id * 1024 + site_id).

namespace stack {
class Query {
 public:
  static PreparedStatement prepareQ1(proteus::QueryShaper &morph);
  static PreparedStatement prepareQ4(proteus::QueryShaper &morph);
  static PreparedStatement prepareQ6(proteus::QueryShaper &morph);
  static PreparedStatement prepareQ11(proteus::QueryShaper &morph);
  static PreparedStatement prepareQ12(proteus::QueryShaper &morph);
  static PreparedStatement prepareQ14(proteus::QueryShaper &morph);

  static std::map<std::string,
                  std::function<double(proteus::InputPrefixQueryShaper &)>>
  getStats(size_t SF);
};

}  // namespace stack

#endif /* PROTEUS_STACK_QUERY_HPP */
