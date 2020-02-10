/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
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

#ifndef PLAN_PARSER_HPP_
#define PLAN_PARSER_HPP_

#include <operators/relbuilder-factory.hpp>
#include <operators/relbuilder.hpp>

#include "expression-parser.hpp"
#include "expressions/expressions.hpp"
#include "plan/catalog-parser.hpp"
#include "rapidjson/document.h"

class Operator;
class Plugin;

class PlanExecutor {
 private:
  PlanExecutor(const char *planPath, CatalogParser &cat,
               const char *moduleName = "llvmModule");
  friend class PreparedStatement;

  void *handle;  // FIXME: when can we release the handle ?

 private:
  const char *moduleName;
  CatalogParser &catalogParser;
  vector<Plugin *> activePlugins;
  std::map<size_t, Operator *> splitOps;
  RelBuilderFactory factory;

  [[deprecated]] ParallelContext *ctx;
  void parsePlan(const rapidjson::Document &doc, bool execute = false);
  /* When processing tree root, parent will be nullptr */
  RelBuilder parseOperator(const rapidjson::Value &val);
  expression_t parseExpression(const rapidjson::Value &val,
                               const expressions::InputArgument &arg) {
    return ExpressionParser{catalogParser, arg}.parseExpression(val, ctx);
  }
  ExpressionType *parseExpressionType(const rapidjson::Value &val,
                                      const expressions::InputArgument &arg) {
    return ExpressionParser{catalogParser, arg}.parseExpressionType(val);
  }
  RecordAttribute *parseRecordAttr(const rapidjson::Value &val,
                                   const expressions::InputArgument &arg,
                                   const ExpressionType *defaultType = nullptr,
                                   int defaultAttrNo = -1) {
    return ExpressionParser{catalogParser, arg}.parseRecordAttr(
        val, defaultType, defaultAttrNo);
  }
  Monoid parseAccumulator(const char *acc,
                          const expressions::InputArgument &arg) {
    return ExpressionParser{catalogParser, arg}.parseAccumulator(acc);
  }

  Plugin *parsePlugin(const rapidjson::Value &val);

  void compileAndLoad();

  void cleanUp();
};

std::ostream &operator<<(std::ostream &out, const rapidjson::Value &val);

#endif /* PLAN_PARSER_HPP_ */
