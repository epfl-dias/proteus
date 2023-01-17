/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2020
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

#include <ssb/query.hpp>

std::map<std::string, std::function<double(proteus::InputPrefixQueryShaper&)>>
ssb::Query::getStats(size_t SF) {
  return {{"sf", [SF](auto&) { return SF; }},
          {"date", [](auto&) { return 2556; }},
          {"customer", [](auto& s) { return s.sf() * 30'000; }},
          {"supplier", [](auto& s) { return s.sf() * 2'000; }},
          {"part",
           [](auto& s) {
             return 200'000 * std::ceil(1 + std::log2((double)s.sf()));
           }},
          {"lineorder", [](auto& s) { return s.sf() * 6'000'000; }}};
}
