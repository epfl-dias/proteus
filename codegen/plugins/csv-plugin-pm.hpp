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

#include "plugins/plugins.hpp"
#include "util/atois.hpp"
#include "util/caching.hpp"

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
            vector<RecordAttribute *> whichFields, int lineHint, int policy,
            bool stringBrackets = true);
  CSVPlugin(Context *const context, string &fname, RecordType &rec,
            vector<RecordAttribute *> whichFields, char delimInner,
            int lineHint, int policy, bool stringBrackets = true);
  /* PM Ready */
  CSVPlugin(Context *const context, string &fname, RecordType &rec,
            vector<RecordAttribute *> whichFields, char delimInner,
            int lineHint, int policy, size_t *newlines, short **offsets,
            bool stringBrackets = true);
  ~CSVPlugin();
  virtual string &getName() { return fname; }
  void init();
  void generate(const ::Operator &producer);
  void finish();
  virtual ProteusValueMemory readPath(string activeRelation, Bindings bindings,
                                      const char *pathVar,
                                      RecordAttribute attr);
  virtual ProteusValueMemory readValue(ProteusValueMemory mem_value,
                                       const ExpressionType *type);
  virtual ProteusValue readCachedValue(CacheInfo info,
                                       const OperatorState &currState);
  virtual ProteusValue readCachedValue(
      CacheInfo info, const map<RecordAttribute, ProteusValueMemory> &bindings);

  virtual ProteusValue hashValue(ProteusValueMemory mem_value,
                                 const ExpressionType *type);
  virtual ProteusValue hashValueEager(ProteusValue value,
                                      const ExpressionType *type);

  virtual void flushTuple(ProteusValueMemory mem_value, llvm::Value *fileName) {
    string error_msg = "[CSVPlugin: ] Functionality not supported yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushValue(ProteusValueMemory mem_value,
                          const ExpressionType *type, llvm::Value *fileName);
  virtual void flushValueEager(ProteusValue value, const ExpressionType *type,
                               llvm::Value *fileName);

  virtual ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) {
    string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                         ProteusValueMemory mem_currentChild) {
    string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) {
    string error_msg = "[CSVPlugin: ] CSV files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual llvm::Value *getValueSize(ProteusValueMemory mem_value,
                                    const ExpressionType *type);

  /* Export PM */
  /* XXX I think it's the 'Caching Service' that should
   * be making the PM available later on */
  short **getOffsetsPM() { return pm; }
  size_t *getNewlinesPM() { return newlines; }
  //    virtual typeID getOIDSize() {
  //        return INT;
  //    }
  virtual ExpressionType *getOIDType() { return new IntType(); }

  virtual PluginType getPluginType() { return PGCSV; }

  virtual void flushBeginList(llvm::Value *fileName) {}

  virtual void flushBeginBag(llvm::Value *fileName) {
    string error_msg = "[CSVPlugin: ] CSV files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushBeginSet(llvm::Value *fileName) {
    string error_msg = "[CSVPlugin: ] CSV files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndList(llvm::Value *fileName) {}

  virtual void flushEndBag(llvm::Value *fileName) {
    string error_msg = "[CSVPlugin: ] CSV files do not contain BAGs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndSet(llvm::Value *fileName) {
    string error_msg = "[CSVPlugin: ] CSV files do not contain SETs";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushDelim(llvm::Value *fileName, int depth) {
    llvm::Function *flushFunc = context->getFunction("flushChar");
    vector<llvm::Value *> ArgsV;
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

  virtual void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                          int depth) {
    llvm::Function *flushFunc = context->getFunction("flushDelim");
    vector<llvm::Value *> ArgsV;
    ArgsV.push_back(resultCtr);
    // XXX JSON-specific -> Serializer business to differentiate
    ArgsV.push_back(context->createInt8((depth == 0) ? '\n' : ','));
    ArgsV.push_back(fileName);
    context->getBuilder()->CreateCall(flushFunc, ArgsV);
  }

 private:
  string fname;
  off_t fsize;
  int fd;
  char *buf;
  // Schema info provided
  RecordType &rec;
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

  // Generates a for loop that performs the file scan
  // No assumption on auxiliary structures yet
  void scanAndPopulatePM(const ::Operator &producer);
  void scanPM(const ::Operator &producer);
};

}  // namespace pm
