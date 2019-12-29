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

#ifndef PROTEUS_CH100W_QUERY_HPP
#define PROTEUS_CH100W_QUERY_HPP

#include <plan/prepared-statement.hpp>

namespace ch100w {
class Query {
 public:
  PreparedStatement prepare01(bool memmv);
  PreparedStatement prepare06(bool memmv);

  std::vector<PreparedStatement> prepareAll(bool memmv) {
    return {
        prepare01(memmv), prepare06(memmv)
        // EOQ
    };
  }
};
};  // namespace ch100w

#endif /* PROTEUS_CH100W_QUERY_HPP */
