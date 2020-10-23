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

#ifndef PROTEUS_PLAN_PARSER_HPP
#define PROTEUS_PLAN_PARSER_HPP

#include <olap/operators/relbuilder-factory.hpp>
#include <olap/plan/catalog-parser.hpp>
#include <span>

namespace proteus {

class PlanParser {
 protected:
  /*
   * RelBuilder factory to be used when requesting new RelBuilders.
   */
  virtual RelBuilderFactory getRelBuilderFactory(std::string name) {
    return RelBuilderFactory{std::move(name)};
  }

  virtual RelBuilder getBuilder(std::string name) {
    return getRelBuilderFactory(std::move(name)).getBuilder();
  }

  virtual CatalogParser &getCatalog() { return CatalogParser::getInstance(); }

 public:
  virtual RelBuilder parse(const std::span<const std::byte> &plan,
                           std::string query_name) = 0;

  virtual ~PlanParser() = default;
};
}  // namespace proteus

#endif /* PROTEUS_PLAN_PARSER_HPP */
