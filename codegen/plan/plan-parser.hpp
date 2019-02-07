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

#include "codegen/util/parallel-context.hpp"
#include "common/common.hpp"
#include "expressions/binary-operators.hpp"
#include "expressions/expressions-hasher.hpp"
#include "expressions/expressions.hpp"
#include "operators/materializer-expr.hpp"
#include "operators/operators.hpp"
#include "plugins/binary-col-plugin.hpp"
#include "plugins/binary-row-plugin.hpp"
#include "plugins/csv-plugin-pm.hpp"
#include "plugins/json-plugin.hpp"
#include "util/caching.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/stringbuffer.h"

using namespace rapidjson;
using namespace std;

typedef struct InputInfo {
  string path;
  ExpressionType *exprType;
  // Used by materializing operations
  ExpressionType *oidType;
} InputInfo;

class CatalogParser;

class ExpressionParser {
  CatalogParser &catalogParser;

 public:
  ExpressionParser(CatalogParser &catalogParser)
      : catalogParser(catalogParser){};
  expression_t parseExpression(const rapidjson::Value &val, Context *ctx);
  ExpressionType *parseExpressionType(const rapidjson::Value &val);
  RecordAttribute *parseRecordAttr(const rapidjson::Value &val,
                                   const ExpressionType *defaultType = NULL);
  Monoid parseAccumulator(const char *acc);

 private:
  expression_t parseExpressionWithoutRegistering(const rapidjson::Value &val,
                                                 Context *ctx);
  expressions::extract_unit parseUnitRange(std::string range, Context *ctx);
  RecordType *getRecordType(string relName);
  const RecordAttribute *getAttribute(string relName, string attrName);
};

class CatalogParser {
  ParallelContext *context;

 public:
  CatalogParser(const char *catalogPath, ParallelContext *context = NULL);

  InputInfo *getInputInfoIfKnown(string inputName) {
    map<string, InputInfo *>::iterator it;
    it = inputs.find(inputName);
    if (it == inputs.end()) return NULL;
    return it->second;
  }

  InputInfo *getInputInfo(string inputName) {
    InputInfo *ret = getInputInfoIfKnown(inputName);
    if (ret) return ret;

    string err = string("Unknown Input: ") + inputName;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }

  InputInfo *getOrCreateInputInfo(string inputName);

  void setInputInfo(string inputName, InputInfo *info) {
    inputs[inputName] = info;
  }

 private:
  void parseCatalogFile(std::string file);
  void parseDir(std::string dir);

  ExpressionParser exprParser;
  map<string, InputInfo *> inputs;
};

class PlanExecutor {
 public:
  PlanExecutor(const char *planPath, CatalogParser &cat,
               const char *moduleName = "llvmModule");
  PlanExecutor(const char *planPath, CatalogParser &cat, const char *moduleName,
               Context *ctx);

 private:
  ExpressionParser exprParser;
  const char *__attribute__((unused)) planPath;
  const char *moduleName;
  CatalogParser &catalogParser;
  vector<Plugin *> activePlugins;
  std::map<size_t, Operator *> splitOps;

  Context *ctx;
  void parsePlan(const rapidjson::Document &doc, bool execute = false);
  /* When processing tree root, parent will be NULL */
  Operator *parseOperator(const rapidjson::Value &val);
  expression_t parseExpression(const rapidjson::Value &val) {
    return exprParser.parseExpression(val, ctx);
  }
  ExpressionType *parseExpressionType(const rapidjson::Value &val) {
    return exprParser.parseExpressionType(val);
  }
  RecordAttribute *parseRecordAttr(const rapidjson::Value &val) {
    return exprParser.parseRecordAttr(val);
  }
  Monoid parseAccumulator(const char *acc) {
    return exprParser.parseAccumulator(acc);
  }

  Plugin *parsePlugin(const rapidjson::Value &val);

  void cleanUp();
};

int lookupInDictionary(string s, const rapidjson::Value &val);

#endif /* PLAN_PARSER_HPP_ */
