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

#ifndef JSON_PLUGIN_HPP_
#define JSON_PLUGIN_HPP_

#include "plugins/plugins.hpp"
#include "util/atois.hpp"
#include "util/caching.hpp"
#include "util/catalog.hpp"

//#define DEBUGJSON

//#JSON
#define JSMN_STRICT
//
//#define JSON_TIGHT
#include "jsmn.h"
//#define DEBUGJSMN

namespace jsonPipelined {

struct pmJSON;

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
 *
 * Token struct contents:
 * 0: jsmntype_t type;
 * 1: int start;
 * 2: int end;
 * 3: int size;
 */

/**
 * Assumptions:
 * 1. Each row contains A JSON OBJECT.
 *       This is the format that the majority of (db) vendors require
 *       when working with JSON
 * 2. FIXME Currently constructs PM from scratch, in pipelined fashion.
 *       Need one more plugin (or one more constructor), taking the PM as
 * granted (Actually, we check if a PM is already cached)
 */

/**
 * FIXME offset can reside in a single array, much like newlines in csv_pm ->
 * No need for it to be a part of the 'tupleIdentifier'
 */
class JSONPlugin : public Plugin {
 public:
  /* XXX Do NOT use this constructor with large inputs until realloc() is
   * implemented for lines
   * XXX Assume linehint is NECESSARY to be provided */
  /* Deprecated */
  //    JSONPlugin(Context* const context, string& fname, ExpressionType*
  //    schema);
  JSONPlugin(Context *const context, string fname, ExpressionType *schema,
             size_t linehint = 1000, bool staticSchema = false);
  JSONPlugin(Context *const context, string fname, ExpressionType *schema,
             size_t linehint, jsmntok_t **tokens);
  ~JSONPlugin();
  void init();
  void generate(const Operator &producer);
  void finish();
  string &getName() { return fname; }

  // 1-1 correspondence with 'RecordProjection' expression
  virtual ProteusValueMemory readPath(string activeRelation,
                                      Bindings wrappedBindings,
                                      const char *pathVar,
                                      RecordAttribute attr);
  virtual ProteusValueMemory readPredefinedPath(string activeRelation,
                                                Bindings wrappedBindings,
                                                RecordAttribute attr);
  virtual ProteusValueMemory readValue(ProteusValueMemory mem_value,
                                       const ExpressionType *type);
  virtual ProteusValue readCachedValue(CacheInfo info,
                                       const OperatorState &currState);
  virtual ProteusValue readCachedValue(
      CacheInfo info, const map<RecordAttribute, ProteusValueMemory> &bindings);

  virtual ProteusValue hashValue(ProteusValueMemory mem_value,
                                 const ExpressionType *type, Context *context);
  virtual ProteusValue hashValueEager(ProteusValue value,
                                      const ExpressionType *type,
                                      Context *context);

  /**
   * XXX VERY strong JSON-specific assumption (pretty much hard-coding) that we
   * can just grab a chunk of the input and flush it w/o caring what is the
   * final serialization format
   *
   * Both also assume that input is an OID (complex one)
   */
  virtual void flushTuple(ProteusValueMemory mem_value, llvm::Value *fileName) {
    flushChunk(mem_value, fileName);
  }
  virtual void flushValue(ProteusValueMemory mem_value,
                          const ExpressionType *type, llvm::Value *fileName) {
    if (type->getTypeID() != DSTRING)
      flushChunk(mem_value, fileName);
    else
      flushDString(mem_value, type, fileName);
  }
  virtual void flushValueEager(ProteusValue value, const ExpressionType *type,
                               llvm::Value *fileName);
  void flushChunk(ProteusValueMemory mem_value, llvm::Value *fileName);

  virtual llvm::Value *getValueSize(ProteusValueMemory mem_value,
                                    const ExpressionType *type);

  // Used by unnest
  virtual ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentTokenNo);
  virtual ProteusValue collectionHasNext(ProteusValue parentTokenId,
                                         ProteusValueMemory mem_currentTokenId);
  virtual ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentToken);

  void scanObjects(const Operator &producer, llvm::Function *debug);

  //    virtual typeID getOIDSize() { return INT; }
  virtual ExpressionType *getOIDType() {
    Int64Type *int64Type = new Int64Type();

    string field1 = string("offset");
    string field2 = string("rowId");
    string field3 = string("tokenNo");

    RecordAttribute *attr1 = new RecordAttribute(1, fname, field1, int64Type);
    RecordAttribute *attr2 = new RecordAttribute(2, fname, field2, int64Type);
    RecordAttribute *attr3 = new RecordAttribute(3, fname, field3, int64Type);
    list<RecordAttribute *> atts = list<RecordAttribute *>();
    atts.push_back(attr1);
    atts.push_back(attr2);
    atts.push_back(attr3);
    RecordType *inner = new RecordType(atts);
    return inner;
  }

  virtual PluginType getPluginType() { return PGJSON; }

  jsmntok_t **getTokens() { return tokens; }

  //    void freeTokens() {
  //        for(int i = 0; i < lines; i++)    {
  //            free(tokens[i]);
  //        }
  //        free(tokens);
  //    }

  virtual void flushBeginList(llvm::Value *fileName) {
    auto flushFunc = context->getFunction("flushChar");
    vector<llvm::Value *> ArgsV{context->createInt8('['), fileName};
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
    auto flushFunc = context->getFunction("flushChar");
    vector<llvm::Value *> ArgsV{context->createInt8(']'), fileName};
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
    auto flushFunc = context->getFunction("flushChar");
    vector<llvm::Value *> ArgsV{context->createInt8(','), fileName};
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  virtual void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                          int depth) {
    auto flushFunc = context->getFunction("flushDelim");
    // XXX JSON-specific -> Serializer business to differentiate
    vector<llvm::Value *> ArgsV{resultCtr, context->createInt8(','), fileName};
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  virtual RecordType getRowType() const {
    return {dynamic_cast<const RecordType &>(
        dynamic_cast<CollectionType &>(*schema).getNestedType())};
  }

  virtual void flushOutput(llvm::Value *fileName) {
    llvm::Function *flushFunc = context->getFunction("flushOutput");
    vector<llvm::Value *> ArgsV;
    // Start 'array'
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

 private:
  string fname;
  size_t fsize;
  int fd;
  const char *buf;
  bool staticSchema;

  llvm::StructType *tokenType;

  /* Specify whether the tokens array will be provided to the PG */
  bool cache;
  /* Specify whether the newlines array will be provided to the PG */
  bool cacheNewlines;
  /* 1-D array of tokens PER ROW => 2D */
  /* 1-D array of newlines offsets per row */
  jsmntok_t **tokens;
  char *tokenBuf;
  size_t *newLines;
  llvm::AllocaInst *mem_tokenArray;
  llvm::AllocaInst *mem_newlineArray;

  /* Start with token value - increase when needed */
  size_t lines;

  // Code-generation-related
  // Used to store memory positions of offset, buf and filesize in the generated
  // code
  map<string, llvm::AllocaInst *> NamedValuesJSON;
  Context *const context;

  // Assumption (1) applies here
  ExpressionType *schema;

  /* Remember: token != OID */
  llvm::StructType *getOIDLLVMType() {
    llvm::LLVMContext &llvmContext = context->getLLVMContext();
    llvm::Type *int64Type = llvm::Type::getInt64Ty(llvmContext);
    vector<llvm::Type *> tokenIdMembers;
    tokenIdMembers.push_back(int64Type);
    tokenIdMembers.push_back(int64Type);
    tokenIdMembers.push_back(int64Type);
    return llvm::StructType::get(context->getLLVMContext(), tokenIdMembers);
  }

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
  ProteusValueMemory readPathInternal(ProteusValueMemory mem_parentTokenId,
                                      const char *pathVar);

  void constructPlugin();
  void createPM();
  void reusePM(pmJSON *pm);
  void initPM();
  void loadPMfromDisk(const char *pmPath, struct stat &pmStatBuffer);

  virtual void flushDString(ProteusValueMemory mem_value,
                            const ExpressionType *type, llvm::Value *fileName);

  llvm::Value *getStart(llvm::Value *jsmnToken);
  llvm::Value *getEnd(llvm::Value *jsmnToken);
};

}  // namespace jsonPipelined
#endif /* JSON_PLUGIN_HPP_ */
