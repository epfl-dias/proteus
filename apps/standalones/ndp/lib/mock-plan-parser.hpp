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

#ifndef PROTEUS_MOCK_PLAN_PARSER_HPP
#define PROTEUS_MOCK_PLAN_PARSER_HPP

#include <plan-parser/plan-parser.hpp>

namespace proteus {

class MockPlanParser : public PlanParser {
 public:
  RelBuilder parse(const std::span<const std::byte> &plan,
                   std::string query_name) override {
    /*
     * In the full system, interprets the plan (which is encoded as a JSON)
     * to decide how to call the RelBuilder.
     *
     * Instead, here we provide example plans to showcase NDP's expressiveness
     */
    return getBuilder(query_name)
        .scan("inputs/ssbm100/date.csv", {"d_datekey", "d_year"}, getCatalog(),
              pg{"distributed-block"})
        .unpack()
        .reduce(
            [&](const auto &arg) -> std::vector<expression_t> {
              return {arg["d_datekey"] + arg["d_year"]};
            },
            {SUM});
  }
};
}  // namespace proteus

#endif /* PROTEUS_MOCK_PLAN_PARSER_HPP */
