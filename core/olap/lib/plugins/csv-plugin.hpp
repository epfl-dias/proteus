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

#include "lib/util/atois.hpp"
#include "olap/plugins/plugins.hpp"

class CSVPlugin : public Plugin {
 public:
  /**
   * Fully eager CSV Plugin. Will probably include another one that only outputs
   * 'tuple metadata' (starting and ending positions).
   *
   * Compared to the GO executor, we need more info to provide to the 'CSV
   * helper' that will be responsible for parsing etc. This information will end
   * up being attached in the query plan
   */
  CSVPlugin(Context *const context, std::string &fname, RecordType &rec_,
            std::vector<RecordAttribute *> &whichFields);
  ~CSVPlugin() override;
  std::string &getName() override { return fname; }
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
    std::string error_msg = "[CSVPlugin: ] No caching support yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValue readCachedValue(
      CacheInfo info,
      const map<RecordAttribute, ProteusValueMemory> &bindings) {
    std::string error_msg = "[CSVPlugin: ] No caching support yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  ProteusValue hashValue(ProteusValueMemory mem_value,
                         const ExpressionType *type, Context *context) override;
  ProteusValue hashValueEager(ProteusValue value, const ExpressionType *type,
                              Context *context) override;

  void flushTuple(ProteusValueMemory mem_value,
                  llvm::Value *fileName) override {
    std::string error_msg = "[CSVPlugin: ] Functionality not supported yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(ProteusValueMemory mem_value, const ExpressionType *type,
                  llvm::Value *fileName) override;
  void flushValueEager(ProteusValue value, const ExpressionType *type,
                       llvm::Value *fileName) override;

  ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) override {
    std::string error_msg =
        "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                 ProteusValueMemory mem_currentChild) override {
    std::string error_msg =
        "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) override {
    std::string error_msg =
        "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  llvm::Value *getValueSize(ProteusValueMemory mem_value,
                            const ExpressionType *type,
                            ParallelContext *context) override;

  //    virtual typeID getOIDSize() { return INT64; }

  ExpressionType *getOIDType() override { return new Int64Type(); }

  PluginType getPluginType() override { return PGCSV; }

  void flushBeginList(llvm::Value *fileName) override {}

  void flushBeginBag(llvm::Value *fileName) override {
    std::string error_msg = "[CSVPlugin: ] CSV files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginSet(llvm::Value *fileName) override {
    std::string error_msg = "[CSVPlugin: ] CSV files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndList(llvm::Value *fileName) override {}

  void flushEndBag(llvm::Value *fileName) override {
    std::string error_msg = "[CSVPlugin: ] CSV files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndSet(llvm::Value *fileName) override {
    std::string error_msg = "[CSVPlugin: ] CSV files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *fileName, int depth) override {
    llvm::Function *flushFunc = context->getFunction("flushChar");
    std::vector<llvm::Value *> ArgsV;
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                  int depth) override {
    llvm::Function *flushFunc = context->getFunction("flushDelim");
    std::vector<llvm::Value *> ArgsV;
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
  std::string fname;
  off_t fsize;
  int fd;
  char *buf;
  // XXX Will remove as soon as skip(), skipDelim() etc become deprecated
  off_t pos;

  // Schema info provided
  RecordType rec;
  std::vector<RecordAttribute *> wantedFields;

  /**
   * Code-generation-related
   */
  // Used to store memory positions of offset, buf and filesize in the generated
  // code
  std::map<std::string, llvm::AllocaInst *> NamedValuesCSV;
  Context *const context;

  const char *posVar;    // = "offset";
  const char *bufVar;    // = "fileBuffer";
  const char *fsizeVar;  // = "fileSize";

  void skip();
  inline size_t skipDelim(size_t pos, char *buf, char delim);
  int readAsInt();
  int eof();

  // Used to generate code
  void skipDelimLLVM(llvm::Value *delim, llvm::Function *debugChar,
                     llvm::Function *debugInt);
  void skipLLVM();
  void skipToEndLLVM();
  void getFieldEndLLVM();
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
  void readAsBooleanLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);

  // Generates a for loop that performs the file scan
  // No assumption on auxiliary structures yet
  void scanCSV(const Operator &producer, llvm::Function *debug);
};
