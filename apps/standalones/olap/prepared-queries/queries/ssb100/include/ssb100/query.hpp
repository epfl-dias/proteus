/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2019
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

#ifndef PROTEUS_SSB100_QUERY_HPP
#define PROTEUS_SSB100_QUERY_HPP

#include <olap/plan/prepared-statement.hpp>
#include <query-shaping/query-shaper.hpp>

namespace ssb100 {
class Query {
 public:
  static PreparedStatement prepare11(proteus::QueryShaper &morph);
  static PreparedStatement prepare12(proteus::QueryShaper &morph);
  static PreparedStatement prepare13(proteus::QueryShaper &morph);
  static PreparedStatement prepare21(proteus::QueryShaper &morph);
  static PreparedStatement prepare22(proteus::QueryShaper &morph);
  static PreparedStatement prepare23(proteus::QueryShaper &morph);
  static PreparedStatement prepare31(proteus::QueryShaper &morph);
  static PreparedStatement prepare32(proteus::QueryShaper &morph);
  static PreparedStatement prepare33(proteus::QueryShaper &morph);
  static PreparedStatement prepare34(proteus::QueryShaper &morph);
  static PreparedStatement prepare41(proteus::QueryShaper &morph);
  static PreparedStatement prepare42(proteus::QueryShaper &morph);
  static PreparedStatement prepare43(proteus::QueryShaper &morph);

  static std::vector<PreparedStatement> prepareAll(bool memmv) {
    proteus::QueryShaperControlMoves shaper{memmv};
    return {
        // Q1.*
        prepare11(shaper), prepare12(shaper), prepare13(shaper),
        // Q2.*
        prepare21(shaper), prepare22(shaper), prepare23(shaper),
        // Q3.*
        prepare31(shaper), prepare32(shaper), prepare33(shaper),
        prepare34(shaper),
        // Q4.*
        prepare41(shaper), prepare42(shaper), prepare43(shaper)
        // EOQ
    };
  }
};
};  // namespace ssb100

#endif /* PROTEUS_SSB100_QUERY_HPP */
