/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2024
        Data Intensive Applications and Systems Laboratory (DIAS)
                Ecole Polytechnique Federale de Lausanne

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

#ifndef PROTEUS_TPCH_ALIASES_HPP
#define PROTEUS_TPCH_ALIASES_HPP

#include <olap-perf-util/benchmark-aliases.hpp>
#include <tpch/query.hpp>

#define FOR_TPCH_QUERY(DO)        \
  DO(Q1, tpch::Query::prepare01); \
  DO(Q3, tpch::Query::prepare03); \
  DO(Q5, tpch::Query::prepare05); \
  DO(Q6, tpch::Query::prepare06); \
  DO(Q12, tpch::Query::prepare12); \
  DO(Q14, tpch::Query::prepare14); \
  DO(Q19, tpch::Query::prepare19);

namespace TPCH100 {
constexpr size_t SF{100};
#define QALIAS(name, prepFunction) QALIASSF(name, prepFunction, TPCH100::SF)
FOR_TPCH_QUERY(QALIAS)
#undef QALIAS
}  // namespace TPCH100

#endif  // PROTEUS_TPCH_ALIASES_HPP
