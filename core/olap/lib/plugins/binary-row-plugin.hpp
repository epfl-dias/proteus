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

#ifndef BINARY_ROW_PLUGIN_HPP_
#define BINARY_ROW_PLUGIN_HPP_

#include "olap/plugins/plugins.hpp"

class Context;

// XXX Tmp Assumption: String are of length 5!
class BinaryRowPlugin : public Plugin {
 public:
  /**
   * Plugin for binary files, organized in tabular format
   */
  BinaryRowPlugin(Context *const context, string &fname, RecordType &rec_,
                  vector<RecordAttribute *> &whichFields);
  ~BinaryRowPlugin() override;
  string &getName() override { return fname; }
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
    string error_msg =
        "[BinaryRowPlugin: ] No caching support should be needed";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValue readCachedValue(
      CacheInfo info,
      const map<RecordAttribute, ProteusValueMemory> &bindings) {
    string error_msg =
        "[BinaryRowPlugin: ] No caching support should be needed";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  ProteusValue hashValue(ProteusValueMemory mem_value,
                         const ExpressionType *type, Context *context) override;
  ProteusValue hashValueEager(ProteusValue value, const ExpressionType *type,
                              Context *context) override;

  ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) override {
    string error_msg =
        "[BinaryRowPlugin: ] Binary row files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                 ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryRowPlugin: ] Binary row files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryRowPlugin: ] Binary row files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushTuple(ProteusValueMemory mem_value,
                  llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(ProteusValueMemory mem_value, const ExpressionType *type,
                  llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValueEager(ProteusValue value, const ExpressionType *type,
                       llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushChunk(ProteusValueMemory mem_value, llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginList(llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginBag(llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginSet(llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndList(llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndBag(llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndSet(llvm::Value *fileName) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *fileName, int depth) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                  int depth) override {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
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

  llvm::Value *getValueSize(ProteusValueMemory mem_value,
                            const ExpressionType *type,
                            ParallelContext *context) override;
  //    virtual typeID getOIDSize() { return INT; }

  ExpressionType *getOIDType() override { return new IntType(); }

  PluginType getPluginType() override { return PGBINARY; }

  RecordType getRowType() const override { return {wantedFields}; }

 private:
  string fname;
  off_t fsize;
  int fd;
  char *buf;

  // Schema info provided
  RecordType rec;
  std::vector<RecordAttribute *> wantedFields;

  /**
   * Code-generation-related
   */
  // Used to store memory positions of offset, buf and filesize in the generated
  // code
  std::map<std::string, llvm::AllocaInst *> NamedValuesBinaryRow;
  Context *const context;

  const char *posVar;    // = "offset";
  const char *bufVar;    // = "fileBuffer";
  const char *fsizeVar;  // = "fileSize";

  // Used to generate code
  void skipLLVM(llvm::Value *offset);
  void readAsIntLLVM(RecordAttribute attName,
                     std::map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsStringLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsFloatLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsBooleanLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
};

#endif /* BINARY_ROW_PLUGIN_HPP_ */
