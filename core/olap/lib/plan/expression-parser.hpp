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

#ifndef PROTEUS_EXPRESSION_PARSER_HPP
#define PROTEUS_EXPRESSION_PARSER_HPP

#include "olap/expressions/expressions.hpp"
#include "olap/plan/catalog-parser.hpp"
#include "rapidjson/document.h"

class ExpressionParser {
  CatalogParser &catalogParser;
  const expressions::InputArgument &arg;

 public:
  ExpressionParser(CatalogParser &catalogParser,
                   const expressions::InputArgument &arg)
      : catalogParser(catalogParser), arg(arg) {}
  expression_t parseExpression(const rapidjson::Value &val,
                               ParallelContext *ctx);
  ExpressionType *parseExpressionType(const rapidjson::Value &val);
  RecordAttribute *parseRecordAttr(const rapidjson::Value &val,
                                   std::string relName,
                                   const ExpressionType *defaultType = nullptr,
                                   int defaultAttrNo = -1);
  RecordAttribute *parseRecordAttr(const rapidjson::Value &val,
                                   const ExpressionType *defaultType = nullptr,
                                   int defaultAttrNo = -1);
  Monoid parseAccumulator(const char *acc);

 private:
  expression_t parseExpressionWithoutRegistering(const rapidjson::Value &val,
                                                 ParallelContext *ctx);
  expressions::extract_unit parseUnitRange(std::string range,
                                           ParallelContext *ctx);
  RecordType *getRecordType(std::string relName, bool createIfNeeded = true);
  const RecordAttribute *getAttribute(std::string relName, std::string attrName,
                                      bool createIfNeeded = true);
};

#endif /* PROTEUS_EXPRESSION_PARSER_HPP */
