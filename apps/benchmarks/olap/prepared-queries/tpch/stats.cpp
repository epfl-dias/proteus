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

// Per-relation row-count hints, used by the QueryShaper to size operators
// (hash tables, mem-move buffers, ...). Cardinalities follow the TPC-H spec:
//   lineitem ~ SF * 6,000,000   orders   ~ SF * 1,500,000
//   partsupp ~ SF *   800,000   customer ~ SF *   150,000
//   part     ~ SF *   200,000   supplier ~ SF *    10,000
//   nation   = 25               region   = 5
std::map<std::string, std::function<double(proteus::InputPrefixQueryShaper&)>>
tpch::Query::getStats(size_t SF) {
  return {{"sf", [SF](auto&) { return SF; }},
          {"region", [](auto&) { return 5; }},
          {"nation", [](auto&) { return 25; }},
          {"supplier", [](auto& s) { return s.sf() * 10'000; }},
          {"customer", [](auto& s) { return s.sf() * 150'000; }},
          {"part", [](auto& s) { return s.sf() * 200'000; }},
          {"partsupp", [](auto& s) { return s.sf() * 800'000; }},
          {"orders", [](auto& s) { return s.sf() * 1'500'000; }},
          {"lineitem", [](auto& s) { return s.sf() * 6'000'000; }}};
}
