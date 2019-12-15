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

#include <plan/prepared-statement.hpp>

namespace ssb100 {
class Query {
 public:
  PreparedStatement prepare11(bool memmv);
  PreparedStatement prepare12(bool memmv);
  PreparedStatement prepare13(bool memmv);
  PreparedStatement prepare21(bool memmv);
  PreparedStatement prepare22(bool memmv);
  PreparedStatement prepare23(bool memmv);
  PreparedStatement prepare31(bool memmv);
  PreparedStatement prepare32(bool memmv);
  PreparedStatement prepare33(bool memmv);
  PreparedStatement prepare34(bool memmv);
  PreparedStatement prepare41(bool memmv);
  PreparedStatement prepare42(bool memmv);
  PreparedStatement prepare43(bool memmv);

  std::vector<PreparedStatement> prepareAll(bool memmv) {
    return {
        // Q1.*
        prepare11(memmv), prepare12(memmv), prepare13(memmv),
        // Q2.*
        prepare21(memmv), prepare22(memmv), prepare23(memmv),
        // Q3.*
        prepare31(memmv), prepare32(memmv), prepare33(memmv), prepare34(memmv),
        // Q4.*
        prepare41(memmv), prepare42(memmv), prepare43(memmv)
        // EOQ
    };
  }
};
};  // namespace ssb100

#endif /* PROTEUS_SSB100_QUERY_HPP */
