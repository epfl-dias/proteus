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

#ifndef JSON_JSMN_PLUGIN_HPP_
#define JSON_JSMN_PLUGIN_HPP_

#include "plugins/plugins.hpp"
#include "util/catalog.hpp"
#include "util/context.hpp"

//#JSON
#define JSMN_STRICT
//
//#define JSON_TIGHT
#include "jsmn.h"
//#define DEBUGJSMN

namespace jsmn {

/**
 * JSON's basic types are:
 * Number: A signed decimal number that may contain a fractional part and may
 * use exponential E notation. JSON does not allow non-numbers like NaN, nor
 * does it make any distinction between integer and floating-point.
 *
 * String: A sequence of zero or more Unicode characters.
 *         Strings are delimited with double-quotation marks and support a
 * backslash escaping syntax.
 *
 * Boolean: either of the values true or false
 *
 * Array: An ORDERED LIST of zero or more values, each of which may be of any
 * type. Arrays use square bracket notation with elements being comma-separated.
 *
 * Object: An UNORDERED ASSOCIATIVE ARRAY (name/value pairs).
 *         Objects are delimited with curly brackets and use commas to separate
 * each pair. All keys must be strings and should be DISTINCT from each other
 * within that object.
 *
 * null — An empty value, using the word null
 */

/**
 * Assumptions:
 * 1. Outermost element is either an object ("List") holding identical elements,
 *       or an array of same logic. The former is handled by the parser, but is
 * not strictly valid JSON
 */
class JSONPlugin : public Plugin {
 public:
  JSONPlugin(Context *const context, string &fname, ExpressionType *schema);
  ~JSONPlugin();
  void init() {}
  void generate(const Operator &producer);
  void finish();
  string &getName() { return fname; }

  // 1-1 correspondence with 'RecordProjection' expression
  virtual ProteusValueMemory readPath(string activeRelation,
                                      Bindings wrappedBindings,
                                      const char *pathVar,
                                      RecordAttribute attr);
  virtual ProteusValueMemory readValue(ProteusValueMemory mem_value,
                                       const ExpressionType *type);
  virtual ProteusValue readCachedValue(CacheInfo info,
                                       const OperatorState &currState) {
    string error_msg = "[JSMNPlugin: ] No caching support yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValue readCachedValue(
      CacheInfo info,
      const map<RecordAttribute, ProteusValueMemory> &bindings) {
    string error_msg = "[JSMNPlugin: ] No caching support yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual ProteusValue hashValue(ProteusValueMemory mem_value,
                                 const ExpressionType *type, Context *context);
  virtual ProteusValue hashValueEager(ProteusValue value,
                                      const ExpressionType *type,
                                      Context *context) {
    string error_msg = "[JSMNPlugin: ] No eager haching support yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  /**
   * XXX VERY strong JSON-specific assumption (pretty much hard-coding) that we
   * can just grab a chunk of the input and flush it w/o caring what is the
   * final serialization format
   */
  virtual void flushTuple(ProteusValueMemory mem_value, llvm::Value *fileName) {
    flushChunk(mem_value, fileName);
  }
  virtual void flushValue(ProteusValueMemory mem_value,
                          const ExpressionType *type, llvm::Value *fileName) {
    flushChunk(mem_value, fileName);
  }
  virtual void flushValueEager(ProteusValue value, const ExpressionType *type,
                               llvm::Value *fileName) {
    string error_msg = "[JSMNPlugin: ] No eager (caching) flushing support yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  void flushChunk(ProteusValueMemory mem_value, llvm::Value *fileName);

  virtual llvm::Value *getValueSize(ProteusValueMemory mem_value,
                                    const ExpressionType *type);

  // Used by unnest
  virtual ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentTokenNo);
  virtual ProteusValue collectionHasNext(ProteusValue val_parentTokenNo,
                                         ProteusValueMemory mem_currentTokenNo);
  virtual ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentToken);

  void scanObjects(const Operator &producer, llvm::Function *debug);
  void scanObjectsInterpreted(list<string> path, list<ExpressionType *> types);
  void scanObjectsEagerInterpreted(list<string> path,
                                   list<ExpressionType *> types);
  void unnestObjectsInterpreted(list<string> path);
  void unnestObjectInterpreted(int parentToken);
  int readPathInterpreted(int parentToken, list<string> path);
  void readValueInterpreted(int tokenNo, const ExpressionType *type);
  void readValueEagerInterpreted(int tokenNo, const ExpressionType *type);
  //    virtual typeID getOIDSize() { return INT; }
  virtual ExpressionType *getOIDType() { return new Int64Type(); }

  virtual PluginType getPluginType() { return PGJSON; }

  virtual void flushBeginList(llvm::Value *fileName) {
    llvm::Function *flushFunc = context->getFunction("flushChar");
    std::vector<llvm::Value *> ArgsV;
    // Start 'array'
    ArgsV.push_back(context->createInt8('['));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  virtual void flushBeginBag(llvm::Value *fileName) {
    string error_msg = string("[JSONPlugin]: Not implemented yet");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushBeginSet(llvm::Value *fileName) {
    string error_msg = string("[JSONPlugin]: Not implemented yet");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndList(llvm::Value *fileName) {
    llvm::Function *flushFunc = context->getFunction("flushChar");
    std::vector<llvm::Value *> ArgsV;
    // Start 'array'
    ArgsV.push_back(context->createInt8(']'));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  virtual void flushEndBag(llvm::Value *fileName) {
    string error_msg = string("[JSONPlugin]: Not implemented yet");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndSet(llvm::Value *fileName) {
    string error_msg = string("[JSONPlugin]: Not implemented yet");
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushDelim(llvm::Value *fileName, int depth) {
    llvm::Function *flushFunc = context->getFunction("flushChar");
    std::vector<llvm::Value *> ArgsV;
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8(','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  virtual void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                          int depth) {
    llvm::Function *flushFunc = context->getFunction("flushDelim");
    std::vector<llvm::Value *> ArgsV;
    ArgsV.push_back(resultCtr);
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8(','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

 private:
  string &fname;
  size_t fsize;
  int fd;
  const char *buf;

  // Code-generation-related
  // Used to store memory positions of offset, buf and filesize in the generated
  // code
  map<string, llvm::AllocaInst *> NamedValuesJSON;
  Context *const context;

  // Assumption (1) applies here
  ExpressionType *__attribute__((unused)) schema;
  jsmntok_t *tokens;

  // Cannot implement such a function. Arrays have non-fixed number of values.
  // Similarly, objects don't always have same number of fields
  // int calculateTokensPerItem(const ExpressionType& expr);

  /*
   * Codegen part
   */
  const char *var_buf;
  const char *var_tokenPtr;
  const char *var_tokenOffset;
  const char *var_tokenOffsetHash;  // Needed to guide hashing process
  void skipToEnd();
  ProteusValueMemory readPathInternal(ProteusValueMemory mem_parentTokenNo,
                                      const char *pathVar);
};

}  // namespace jsmn
#endif /* JSON_JSMN_PLUGIN_HPP_ */
