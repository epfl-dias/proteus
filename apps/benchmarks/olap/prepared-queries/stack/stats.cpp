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

#include <stack/query.hpp>

// Per-relation row-count hints, used by the QueryShaper to size operators.
// The Stack dataset has a single fixed scale (July 2008 - September 2019 dump
// of 170 StackExchange sites); SF is accepted for interface compatibility but
// ignored. These are the row counts for the Stack dataset used by the
// evaluation.
std::map<std::string, std::function<double(proteus::InputPrefixQueryShaper&)>>
stack::Query::getStats(size_t SF) {
  return {{"sf", [SF](auto&) { return SF; }},
          {"site", [](auto&) { return 173; }},
          {"tag", [](auto&) { return 186'770; }},
          {"question", [](auto&) { return 12'666'441; }},
          {"answer", [](auto&) { return 6'347'553; }},
          {"tag_question", [](auto&) { return 36'883'819; }},
          {"so_user", [](auto&) { return 21'097'302; }},
          {"badge", [](auto&) { return 51'236'903; }}};
}
