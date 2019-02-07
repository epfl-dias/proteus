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

// XXX Tmp Assumption: String are of length 5!
class BinaryRowPlugin : public Plugin {
 public:
  /**
   * Plugin for binary files, organized in tabular format
   */
  BinaryRowPlugin(Context *const context, string &fname, RecordType &rec_,
                  vector<RecordAttribute *> &whichFields);
  ~BinaryRowPlugin();
  virtual string &getName() { return fname; }
  void init();
  void generate(const Operator &producer);
  void finish();
  virtual ProteusValueMemory readPath(string activeRelation, Bindings bindings,
                                      const char *pathVar,
                                      RecordAttribute attr);
  virtual ProteusValueMemory readValue(ProteusValueMemory mem_value,
                                       const ExpressionType *type);
  virtual ProteusValue readCachedValue(CacheInfo info,
                                       const OperatorState &currState) {
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

  virtual ProteusValue hashValue(ProteusValueMemory mem_value,
                                 const ExpressionType *type);
  virtual ProteusValue hashValueEager(ProteusValue value,
                                      const ExpressionType *type);

  virtual ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) {
    string error_msg =
        "[BinaryRowPlugin: ] Binary row files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                         ProteusValueMemory mem_currentChild) {
    string error_msg =
        "[BinaryRowPlugin: ] Binary row files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) {
    string error_msg =
        "[BinaryRowPlugin: ] Binary row files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushTuple(ProteusValueMemory mem_value, llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushValue(ProteusValueMemory mem_value,
                          const ExpressionType *type, llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushValueEager(ProteusValue value, const ExpressionType *type,
                               llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushChunk(ProteusValueMemory mem_value, llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushBeginList(llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushBeginBag(llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushBeginSet(llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndList(llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndBag(llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndSet(llvm::Value *fileName) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushDelim(llvm::Value *fileName, int depth) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                          int depth) {
    string error_msg = "[BinaryRowPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual llvm::Value *getValueSize(ProteusValueMemory mem_value,
                                    const ExpressionType *type);
  //    virtual typeID getOIDSize() { return INT; }

  virtual ExpressionType *getOIDType() { return new IntType(); }

  virtual PluginType getPluginType() { return PGBINARY; }

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

  // Generates a for loop that performs the file scan
  void scan(const Operator &producer, llvm::Function *f);
};
