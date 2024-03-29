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
#ifndef BINARY_COL_PLUGIN_HPP_
#define BINARY_COL_PLUGIN_HPP_

#include <olap/plugins/plugins.hpp>
#include <storage/storage-manager.hpp>

//#ifdef DEBUG
#define DEBUGBINCOL
//#endif

class Context;

class BinaryColPlugin : public Plugin {
 public:
  /**
   * Plugin for binary files, organized in COLUMNAR format
   * Emulating access to a column store's native storage
   *
   * Support for
   * -> integers
   * -> floats
   * -> chars & booleans
   * -> varchars & dates
   *
   *
   * Conventions / Assumptions:
   * -> filenames have the form fnamePrefix.<fieldName>
   * -> integers, floats, chars and booleans are stored as tight arrays
   * -> varchars are stored using a dictionary.
   *         Assuming the dictionary file is named fnamePrefix.<fieldName>.dict
   * -> dates require more sophisticated serialization (boost?)
   *
   * EACH FILE CONTAINS A size_t COUNTER AS ITS FIRST ENTRY!
   * => Looping does not depend on fsize: it depends on this 'size/length'
   */

  BinaryColPlugin(Context *const context, string fnamePrefix, RecordType rec,
                  vector<RecordAttribute *> &whichFields,
                  bool sizeInFile = true);
  //    BinaryColPlugin(Context* const context, vector<RecordAttribute*>&
  //    whichFields, vector<CacheInfo> whichCaches);
  ~BinaryColPlugin() override;
  string &getName() override { return fnamePrefix; }
  void init() override;
  //    void initCached();
  void generate(const Operator &producer, ParallelContext *context) override;
  void finish() override;
  ProteusValueMemory readPath(string activeRelation, Bindings bindings,
                              const char *pathVar, RecordAttribute attr,
                              ParallelContext *context) override;
  ProteusValueMemory readValue(ProteusValueMemory mem_value,
                               const ExpressionType *type,
                               ParallelContext *context) override;
  ProteusValue readCachedValue(CacheInfo info, const OperatorState &currState,
                               ParallelContext *context) override;
  //    {
  //        string error_msg = "[BinaryColPlugin: ] No caching support should be
  //        needed"; LOG(ERROR) << error_msg; throw runtime_error(error_msg);
  //    }
  virtual ProteusValue readCachedValue(
      CacheInfo info, const map<RecordAttribute, ProteusValueMemory> &bindings);

  ProteusValue hashValue(ProteusValueMemory mem_value,
                         const ExpressionType *type, Context *context) override;
  ProteusValue hashValueEager(ProteusValue value, const ExpressionType *type,
                              Context *context) override;

  ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) override {
    string error_msg =
        "[BinaryColPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                 ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryColPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryColPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushTuple(ProteusValueMemory mem_value,
                  llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(ProteusValueMemory mem_value, const ExpressionType *type,
                  llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValueEager(ProteusValue value, const ExpressionType *type,
                       llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushChunk(ProteusValueMemory mem_value, llvm::Value *fileName) {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  llvm::Value *getValueSize(ProteusValueMemory mem_value,
                            const ExpressionType *type,
                            ParallelContext *context) override;

  ExpressionType *getOIDType() override { return new Int64Type(); }

  PluginType getPluginType() override { return PGBINARY; }

  void flushBeginList(llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginBag(llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginSet(llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndList(llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndBag(llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndSet(llvm::Value *fileName) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *fileName, int depth) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                  int depth) override {
    string error_msg = "[BinaryColPlugin: ] Flush not implemented yet";
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

  RecordType getRowType() const override { return {wantedFields}; }

 private:
  std::vector<FileRequest> files;
  // Schema info provided
  RecordType rec;
  vector<RecordAttribute *> wantedFields;

  /* Used when we treat the col. files as internal caches! */
  bool isCached;
  vector<CacheInfo> whichCaches;

  string fnamePrefix;
  off_t *colFilesize;  // Size of each column
  const void **buf;

  // Mapping attrNumber to
  //    -> file descriptor of its dictionary
  //    -> file size of its dictionary
  //    -> mapped input of its dictionary
  map<int, int> dictionaries;
  map<int, off_t> dictionaryFilesizes;
  // Note: char* buf can be cast to appropriate struct
  // Struct will probably look like { implicit_oid, len, char* }
  map<int, char *> dictionariesBuf;

  /**
   * Code-generation-related
   */
  // Used to store memory positions of offset, buf and filesize in the generated
  // code
  map<string, llvm::AllocaInst *> NamedValuesBinaryCol;
  Context *const context;
  // To be initialized by init(). Dictates # of loops
  llvm::Value *val_size;

  bool sizeInFile;

  const char *posVar;                            // = "offset";
  const char *bufVar;                            // = "fileBuffer";
  const char *__attribute__((unused)) fsizeVar;  // = "fileSize";
  const char *__attribute__((unused)) sizeVar;   // = "size";
  const char *itemCtrVar;                        // = "itemCtr";

  // Used to generate code
  void skipLLVM(RecordAttribute attName, llvm::Value *offset);
  void prepareArray(RecordAttribute attName);

  void nextEntry();
  /* Operate over char* */
  void readAsInt64LLVM(RecordAttribute attName,
                       map<RecordAttribute, ProteusValueMemory> &variables);
  llvm::Value *readAsInt64LLVM(RecordAttribute attName);
  /* Operates over int* */
  void readAsIntLLVM(RecordAttribute attName,
                     map<RecordAttribute, ProteusValueMemory> &variables);
  /* Operates over float* */
  void readAsFloatLLVM(RecordAttribute attName,
                       map<RecordAttribute, ProteusValueMemory> &variables);
  /* Operates over bool* */
  void readAsBooleanLLVM(RecordAttribute attName,
                         map<RecordAttribute, ProteusValueMemory> &variables);
  /* Not (fully) supported yet. Dictionary-based */
  void readAsStringLLVM(RecordAttribute attName,
                        map<RecordAttribute, ProteusValueMemory> &variables);

  // Generates a for loop that performs the file scan
  void scan(const Operator &producer);
};

#endif /* BINARY_COL_PLUGIN_HPP_ */
