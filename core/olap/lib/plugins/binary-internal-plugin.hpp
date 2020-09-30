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

#ifndef BINARY_INTERNAL_PLUGIN_HPP_
#define BINARY_INTERNAL_PLUGIN_HPP_

#include <olap/util/parallel-context.hpp>

#include "lib/util/atois.hpp"
#include "olap/plugins/plugins.hpp"

//#define DEBUGBINCACHE

class BinaryInternalPlugin : public Plugin {
 public:
  /**
   * Plugin to be used for already serialized bindings.
   * Example: After a nesting operator has taken place.
   *
   * Why not RecordType info?
   * -> Because it requires declaring RecordAttributes
   *    -> ..but RecordAttributes require an associated plugin
   *         (chicken - egg prob.)
   */
  BinaryInternalPlugin(Context *const context, string structName);
  /* Radix-related atm.
   * Resembles BinaryRowPg */
  BinaryInternalPlugin(Context *const context, RecordType rec,
                       string structName, vector<RecordAttribute *> whichOIDs,
                       vector<RecordAttribute *> whichFields, CacheInfo info);
  ~BinaryInternalPlugin() override;
  string &getName() override { return structName; }
  void init() override;
  void generate(const Operator &producer, ParallelContext *context) override;
  void finish() override;
  ProteusValueMemory readPath(string activeRelation, Bindings bindings,
                              const char *pathVar, RecordAttribute attr,
                              ParallelContext *context) override;
  ProteusValueMemory readValue(ProteusValueMemory mem_value,
                               const ExpressionType *type,
                               ParallelContext *context) override;
  ProteusValue readCachedValue(CacheInfo info, const OperatorState &currState,
                               ParallelContext *context) override {
    string error_msg = "[BinaryInternalPlugin: ] No caching support applicable";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValue readCachedValue(
      CacheInfo info,
      const map<RecordAttribute, ProteusValueMemory> &bindings) {
    string error_msg = "[BinaryInternalPlugin: ] No caching support applicable";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  ProteusValue hashValue(ProteusValueMemory mem_value,
                         const ExpressionType *type, Context *context) override;
  ProteusValue hashValueEager(ProteusValue value, const ExpressionType *type,
                              Context *context) override;

  void flushTuple(ProteusValueMemory mem_value,
                  llvm::Value *fileName) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Functionality not supported yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(ProteusValueMemory mem_value, const ExpressionType *type,
                  llvm::Value *fileName) override;
  void flushValueEager(ProteusValue value, const ExpressionType *type,
                       llvm::Value *fileName) override;

  ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Functionality not supported yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                 ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Functionality not supported yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Functionality not supported yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  llvm::Value *getValueSize(ProteusValueMemory mem_value,
                            const ExpressionType *type,
                            ParallelContext *context) override;

  ExpressionType *getOIDType() override { return new IntType(); }

  PluginType getPluginType() override { return PGBINARY; }

  void flushBeginList(llvm::Value *fileName) override {}

  void flushBeginBag(llvm::Value *fileName) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Binary-internal files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginSet(llvm::Value *fileName) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Binary-internal files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndList(llvm::Value *fileName) override {}

  void flushEndBag(llvm::Value *fileName) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Binary-internal files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushOutput(llvm::Value *fileName) override {
    llvm::Function *flushFunc = context->getFunction("flushOutput");
    vector<llvm::Value *> ArgsV;
    // Start 'array'
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  void flushEndSet(llvm::Value *fileName) override {
    string error_msg =
        "[BinaryInternalPlugin: ] Binary-internal files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *fileName, int depth) override {
    auto flushFunc = context->getFunction("flushChar");
    vector<llvm::Value *> ArgsV;
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                  int depth) override {
    auto flushFunc = context->getFunction("flushDelim");
    vector<llvm::Value *> ArgsV;
    ArgsV.push_back(resultCtr);
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  RecordType getRowType() const override { return {fields}; }

 private:
  string structName;

  /**
   * Code-generation-related
   */
  Context *const context;
  /* Radix-related atm */
  void scan(const Operator &producer);
  void scanStruct(const Operator &producer);
  //
  RecordType rec;
  // Number of entries, if applicable
  llvm::Value *val_entriesNo;
  /* Necessary if we are to iterate over the internal caches */

  vector<RecordAttribute *> fields;
  vector<RecordAttribute *> OIDs;

  llvm::StructType *payloadType;
  llvm::Value *mem_buffer;
  llvm::Value *val_structBufferPtr;
  /* Binary offset in file */
  llvm::AllocaInst *mem_pos;
  /* Tuple counter */
  llvm::AllocaInst *mem_cnt;

  /* Since we allow looping over cache, we must also extract fields
   * while looping */
  void skipLLVM(llvm::Value *offset);
  void readAsIntLLVM(RecordAttribute attName,
                     map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsInt64LLVM(RecordAttribute attName,
                       map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsStringLLVM(RecordAttribute attName,
                        map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsFloatLLVM(RecordAttribute attName,
                       map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsBooleanLLVM(RecordAttribute attName,
                         map<RecordAttribute, ProteusValueMemory> &variables);
};

#endif
