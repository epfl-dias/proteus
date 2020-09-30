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
#ifndef CSV_PLUGIN_PM_HPP_
#define CSV_PLUGIN_PM_HPP_

#include <olap/util/parallel-context.hpp>

#include "lib/util/atois.hpp"
#include "lib/util/caching.hpp"
#include "olap/plugins/plugins.hpp"

#define DEBUGPM
#undef DEBUGPM

typedef struct pmCSV {
  size_t *newlines;
  short **offsets;
} pmCSV;

namespace pm {

class CSVPlugin : public Plugin {
 public:
  /**
   * Fully eager CSV Plugin.
   * Populates PM.
   *
   * XXX IMPORTANT: FIELDS MUST BE IN ORDER!!!
   */
  CSVPlugin(Context *const context, string &fname, RecordType &rec,
            const vector<RecordAttribute *> &whichFields, int lineHint,
            int policy, bool stringBrackets = true);
  CSVPlugin(Context *const context, string &fname, RecordType &rec,
            vector<RecordAttribute *> whichFields, char delimInner,
            int lineHint, int policy, bool stringBrackets = true,
            bool hasHeader = false);
  /* PM Ready */
  CSVPlugin(Context *const context, string &fname, RecordType &rec,
            vector<RecordAttribute *> whichFields, char delimInner,
            int lineHint, int policy, size_t *newlines, short **offsets,
            bool stringBrackets = true, bool hasHeader = false);
  string &getName() override { return fname; }
  void init() override;
  void generate(const ::Operator &producer, ParallelContext *context) override;
  void finish() override;
  ProteusValueMemory readPath(string activeRelation, Bindings bindings,
                              const char *pathVar, RecordAttribute attr,
                              ParallelContext *context) override;
  ProteusValueMemory readValue(ProteusValueMemory mem_value,
                               const ExpressionType *type,
                               ParallelContext *context) override;
  ProteusValue readCachedValue(CacheInfo info, const OperatorState &currState,
                               ParallelContext *context) override;
  virtual ProteusValue readCachedValue(
      CacheInfo info, const map<RecordAttribute, ProteusValueMemory> &bindings);

  ProteusValue hashValue(ProteusValueMemory mem_value,
                         const ExpressionType *type, Context *context) override;
  ProteusValue hashValueEager(ProteusValue value, const ExpressionType *type,
                              Context *context) override;

  void flushTuple(ProteusValueMemory mem_value,
                  llvm::Value *fileName) override {
    string error_msg = "[CSVPlugin: ] Functionality not supported yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(ProteusValueMemory mem_value, const ExpressionType *type,
                  llvm::Value *fileName) override;
  void flushValueEager(ProteusValue value, const ExpressionType *type,
                       llvm::Value *fileName) override;

  ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) override {
    string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                 ProteusValueMemory mem_currentChild) override {
    string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) override {
    string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  llvm::Value *getValueSize(ProteusValueMemory mem_value,
                            const ExpressionType *type,
                            ParallelContext *context) override;

  /* Export PM */
  /* XXX I think it's the 'Caching Service' that should
   * be making the PM available later on */
  short **getOffsetsPM() { return pm; }
  size_t *getNewlinesPM() { return newlines; }
  //    virtual typeID getOIDSize() {
  //        return INT;
  //    }
  ExpressionType *getOIDType() override { return new IntType(); }

  PluginType getPluginType() override { return PGCSV; }

  void flushBeginList(llvm::Value *fileName) override {}

  void flushBeginBag(llvm::Value *fileName) override {
    string error_msg = "[CSVPlugin: ] CSV files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginSet(llvm::Value *fileName) override {
    string error_msg = "[CSVPlugin: ] CSV files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndList(llvm::Value *fileName) override {}

  void flushEndBag(llvm::Value *fileName) override {
    string error_msg = "[CSVPlugin: ] CSV files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndSet(llvm::Value *fileName) override {
    string error_msg = "[CSVPlugin: ] CSV files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *fileName, int depth) override {
    llvm::Function *flushFunc = context->getFunction("flushChar");
    vector<llvm::Value *> ArgsV;
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                  int depth) override {
    llvm::Function *flushFunc = context->getFunction("flushDelim");
    vector<llvm::Value *> ArgsV;
    ArgsV.push_back(resultCtr);
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  void flushOutput(llvm::Value *fileName) override {
    llvm::Function *flushFunc = context->getFunction("flushOutput");
    vector<llvm::Value *> ArgsV;
    // Start 'array'
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  RecordType getRowType() const override { return wantedFields; }

 private:
  string fname;
  off_t fsize;
  int fd;
  char *buf;
  // Schema info provided
  RecordType rec;
  vector<RecordAttribute *> wantedFields;
  bool stringBrackets;  // Are string literals wrapped in ""?

  /**
   * PM-related
   */
  char delimInner;
  char delimEnd;
  size_t lines;
  int policy;
  /* Indicates whether a PM was provided at construction time*/
  bool hasPM;
  size_t *newlines;
  /* All pm entries are relevant to linesStart!!! */
  short **pm;

  const bool hasHeader;

  llvm::AllocaInst *mem_newlines;
  llvm::AllocaInst *mem_pm;
  llvm::AllocaInst *mem_lineCtr;

  /**
   * Code-generation-related
   */
  // Used to store memory positions of offset, buf and filesize in the generated
  // code
  map<string, llvm::AllocaInst *> NamedValuesCSV;
  Context *const context;

  const char *posVar;    // = "offset";
  const char *bufVar;    // = "fileBuffer";
  const char *fsizeVar;  // = "fileSize";

  // Used to generate code
  void skipDelimLLVM(llvm::Value *delim);
  void skipDelimBackwardsLLVM(llvm::Value *delim);
  void skipLLVM();
  void skipToEndLLVM();
  void getFieldEndLLVM();
  void readField(typeID id, RecordAttribute attName,
                 std::map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsIntLLVM(RecordAttribute attName,
                     std::map<RecordAttribute, ProteusValueMemory> &variables,
                     llvm::Function *atoi_, llvm::Function *debugChar,
                     llvm::Function *debugInt);
  void readAsIntLLVM(RecordAttribute attName,
                     std::map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsFloatLLVM(RecordAttribute attName,
                       std::map<RecordAttribute, ProteusValueMemory> &variables,
                       llvm::Function *atof_, llvm::Function *debugChar,
                       llvm::Function *debugFloat);
  void readAsFloatLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsBooleanLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  /* Assumption: String comes bracketted */
  void readAsStringLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  void readAsDateLLVM(RecordAttribute attName,
                      std::map<RecordAttribute, ProteusValueMemory> &variables);

  // Generates a for loop that performs the file scan
  // No assumption on auxiliary structures yet
  void scanAndPopulatePM(const ::Operator &producer);
  void scanPM(const ::Operator &producer);
};

}  // namespace pm

#endif /* CSV_PLUGIN_PM_HPP_ */
