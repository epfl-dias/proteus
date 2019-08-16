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
#include "rapidjson/document.h"
#include "rapidjson/reader.h"
#include "rapidjson/stringbuffer.h"
#include "util/caching.hpp"
#include "util/functions.hpp"
#include "values/expressionTypes.hpp"

typedef struct InputInfo {
  std::string path;
  ExpressionType *exprType;
  // Used by materializing operations
  ExpressionType *oidType;
} InputInfo;

class CatalogParser;

class ExpressionParser {
  CatalogParser &catalogParser;

 public:
  ExpressionParser(CatalogParser &catalogParser)
      : catalogParser(catalogParser) {}
  expression_t parseExpression(const rapidjson::Value &val, Context *ctx);
  ExpressionType *parseExpressionType(const rapidjson::Value &val);
  RecordAttribute *parseRecordAttr(const rapidjson::Value &val,
                                   const ExpressionType *defaultType = nullptr,
                                   int defaultAttrNo = -1);
  Monoid parseAccumulator(const char *acc);

 private:
  expression_t parseExpressionWithoutRegistering(const rapidjson::Value &val,
                                                 Context *ctx);
  expressions::extract_unit parseUnitRange(std::string range, Context *ctx);
  RecordType *getRecordType(std::string relName, bool createIfNeeded = true);
  const RecordAttribute *getAttribute(std::string relName, std::string attrName,
                                      bool createIfNeeded = true);
};

class CatalogParser {
  ParallelContext *context;

 public:
  CatalogParser(const char *catalogPath, ParallelContext *context = nullptr);

  InputInfo *getInputInfoIfKnown(std::string inputName) {
    map<std::string, InputInfo *>::iterator it;
    it = inputs.find(inputName);
    if (it == inputs.end()) return nullptr;
    return it->second;
  }

  InputInfo *getInputInfo(std::string inputName) {
    InputInfo *ret = getInputInfoIfKnown(inputName);
    if (ret) return ret;

    std::string err = std::string("Unknown Input: ") + inputName;
    LOG(ERROR) << err;
    throw runtime_error(err);
  }

  InputInfo *getOrCreateInputInfo(std::string inputName);

  void setInputInfo(std::string inputName, InputInfo *info) {
    inputs[inputName] = info;
  }

 private:
  void parseCatalogFile(std::string file);
  void parseDir(std::string dir);

  ExpressionParser exprParser;
  map<std::string, InputInfo *> inputs;
};

class PlanExecutor {
 private:
  PlanExecutor(const char *planPath, CatalogParser &cat,
               const char *moduleName = "llvmModule");
  PlanExecutor(const char *planPath, CatalogParser &cat, const char *moduleName,
               Context *ctx);
  friend class PreparedStatement;

  void *handle;  // FIXME: when can we release the handle ?

 private:
  ExpressionParser exprParser;
  const char *__attribute__((unused)) planPath;
  const char *moduleName;
  CatalogParser &catalogParser;
  vector<Plugin *> activePlugins;
  std::map<size_t, Operator *> splitOps;

  Context *ctx;
  void parsePlan(const rapidjson::Document &doc, bool execute = false);
  /* When processing tree root, parent will be nullptr */
  Operator *parseOperator(const rapidjson::Value &val);
  expression_t parseExpression(const rapidjson::Value &val) {
    return exprParser.parseExpression(val, ctx);
  }
  ExpressionType *parseExpressionType(const rapidjson::Value &val) {
    return exprParser.parseExpressionType(val);
  }
  RecordAttribute *parseRecordAttr(const rapidjson::Value &val,
                                   const ExpressionType *defaultType = nullptr,
                                   int defaultAttrNo = -1) {
    return exprParser.parseRecordAttr(val, defaultType, defaultAttrNo);
  }
  Monoid parseAccumulator(const char *acc) {
    return exprParser.parseAccumulator(acc);
  }

  Plugin *parsePlugin(const rapidjson::Value &val);

  void cleanUp();
};

int lookupInDictionary(std::string s, const rapidjson::Value &val);

std::ostream &operator<<(std::ostream &out, const rapidjson::Value &val);

#endif /* PLAN_PARSER_HPP_ */
