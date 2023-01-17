/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2022
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

#ifndef PROTEUS_SSB_ALIASES_HPP
#define PROTEUS_SSB_ALIASES_HPP

#include <olap-perf-util/benchmark-aliases.hpp>
#include <ssb/query.hpp>

#define FOR_SSB_QUERY(DO)          \
  DO(Q1_1, ssb::Query::prepare11); \
  DO(Q1_2, ssb::Query::prepare12); \
  DO(Q1_3, ssb::Query::prepare13); \
  DO(Q2_1, ssb::Query::prepare21); \
  DO(Q2_2, ssb::Query::prepare22); \
  DO(Q2_3, ssb::Query::prepare23); \
  DO(Q3_1, ssb::Query::prepare31); \
  DO(Q3_2, ssb::Query::prepare32); \
  DO(Q3_3, ssb::Query::prepare33); \
  DO(Q3_4, ssb::Query::prepare34); \
  DO(Q4_1, ssb::Query::prepare41); \
  DO(Q4_2, ssb::Query::prepare42); \
  DO(Q4_3, ssb::Query::prepare43);

namespace SSB100 {
constexpr size_t SF{100};
#define QALIAS(name, prepFunction) QALIASSF(name, prepFunction, SSB100::SF)
FOR_SSB_QUERY(QALIAS)
#undef QALIAS
}  // namespace SSB100

namespace SSB1000 {
constexpr size_t SF = 1000;
#define QALIAS(name, prepFunction) QALIASSF(name, prepFunction, SSB1000::SF)
FOR_SSB_QUERY(QALIAS)
#undef QALIAS
}  // namespace SSB1000

#endif  // PROTEUS_SSB_ALIASES_HPP
