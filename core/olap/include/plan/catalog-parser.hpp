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

#ifndef CATALOG_PARSER_HPP_
#define CATALOG_PARSER_HPP_

#include "values/expressionTypes.hpp"

class ParallelContext;

typedef struct InputInfo {
  std::string path;
  ExpressionType *exprType;
  // Used by materializing operations
  ExpressionType *oidType;
} InputInfo;

class CatalogParser {
  ParallelContext *context;

 public:
  CatalogParser(const char *catalogPath, ParallelContext *context = nullptr);

  static CatalogParser &getInstance();

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
  InputInfo *getOrCreateInputInfo(std::string inputName,
                                  ParallelContext *context);

  void setInputInfo(std::string inputName, InputInfo *info) {
    inputs[inputName] = info;
  }

  void registerInput(std::string inputName, ExpressionType *type) {
    auto ii = new InputInfo;
    ii->exprType = type;
    ii->path = inputName;
    ii->oidType = nullptr;
    inputs[inputName] = ii;
  }

  void clear();

 private:
  void parseCatalogFile(std::string file);
  void parseDir(std::string dir);

  map<std::string, InputInfo *> inputs;
};

#endif /* CATALOG_PARSER_HPP_ */
