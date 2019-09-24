/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

#ifndef SCAN_TO_BLOCKS_SM_PLUGIN_HPP_
#define SCAN_TO_BLOCKS_SM_PLUGIN_HPP_

#include <memory>

#include "codegen/util/parallel-context.hpp"
#include "plugins/plugins.hpp"
#include "storage/storage-manager.hpp"

class ScanToBlockSMPlugin : public Plugin {
  /**
   * Plugin for scanning columns on the gpu side.
   *
   * Support for
   * -> integers
   * -> floats
   * -> chars & booleans
   *
   *
   * Conventions / Assumptions:
   * -> integers, floats, chars and booleans are stored as tight arrays
   * -> varchars are not yet supported
   * -> dates require more sophisticated serialization (boost?)
   */
 protected:
  ScanToBlockSMPlugin(ParallelContext *const context, string fnamePrefix,
                      RecordType rec,
                      std::vector<RecordAttribute *> &whichFields, bool load);

 public:
  ScanToBlockSMPlugin(ParallelContext *const context, string fnamePrefix,
                      RecordType rec,
                      std::vector<RecordAttribute *> &whichFields);

  ScanToBlockSMPlugin(ParallelContext *const context, string fnamePrefix,
                      RecordType rec);
  //  ScanToBlockSMPlugin(ParallelContext* const context,
  //  vector<RecordAttribute*>& whichFields, vector<CacheInfo> whichCaches);
  ~ScanToBlockSMPlugin();
  virtual string &getName() { return fnamePrefix; }
  void init();
  //  void initCached();
  void generate(const Operator &producer);
  void finish();
  virtual ProteusValueMemory readPath(string activeRelation, Bindings bindings,
                                      const char *pathVar,
                                      RecordAttribute attr);
  virtual ProteusValueMemory readValue(ProteusValueMemory mem_value,
                                       const ExpressionType *type);
  virtual ProteusValue readCachedValue(CacheInfo info,
                                       const OperatorState &currState);

  virtual ProteusValue hashValue(ProteusValueMemory mem_value,
                                 const ExpressionType *type);
  virtual ProteusValue hashValueEager(ProteusValue value,
                                      const ExpressionType *type);

  virtual ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) {
    string error_msg =
        "[ScanToBlockSMPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                         ProteusValueMemory mem_currentChild) {
    string error_msg =
        "[ScanToBlockSMPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  virtual ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) {
    string error_msg =
        "[ScanToBlockSMPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushTuple(ProteusValueMemory mem_value, llvm::Value *fileName) {
    string error_msg = "[ScanToBlockSMPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushValue(ProteusValueMemory mem_value,
                          const ExpressionType *type, llvm::Value *fileName) {
    string error_msg = "[ScanToBlockSMPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushValueEager(ProteusValue value, const ExpressionType *type,
                               llvm::Value *fileName) {
    string error_msg = "[ScanToBlockSMPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushChunk(ProteusValueMemory mem_value, llvm::Value *fileName) {
    string error_msg = "[ScanToBlockSMPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual llvm::Value *getValueSize(ProteusValueMemory mem_value,
                                    const ExpressionType *type);

  virtual ExpressionType *getOIDType() { return new Int64Type(); }

  virtual PluginType getPluginType() { return PGBINARY; }

  virtual void flushBeginList(llvm::Value *fileName) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushBeginBag(llvm::Value *fileName) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushBeginSet(llvm::Value *fileName) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndList(llvm::Value *fileName) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndBag(llvm::Value *fileName) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushEndSet(llvm::Value *fileName) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushDelim(llvm::Value *fileName, int depth) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  virtual void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                          int depth) {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

 protected:
  void finalize_data();
  virtual RecordType getRowType() const;

  virtual std::pair<llvm::Value *, llvm::Value *> getPartitionSizes() const;
  virtual void freePartitionSizes(llvm::Value *) const;

  virtual llvm::Value *getDataPointersForFile(size_t i) const;
  virtual void freeDataPointersForFile(size_t i, llvm::Value *) const;

  std::vector<RecordAttribute *> wantedFields;
  std::vector<std::vector<mem_file>> wantedFieldsFiles;

 private:
  // Schema info provided
  RecordType rec;
  std::vector<int> wantedFieldsArg_id;
  // llvm::Value *               tupleCnt;
  llvm::Value *blockSize;

  std::vector<size_t> wantedFieldsWidth;

  std::vector<llvm::Type *> parts_array;
  llvm::StructType *parts_arrays_type;

  size_t Nparts;

  /* Used when we treat the col. files as internal caches! */
  std::vector<CacheInfo> whichCaches;

  string fnamePrefix;
  off_t *colFilesize;  // Size of each column
  int *fd;             // One per column
  char **buf;

  // Mapping attrNumber to
  //  -> file descriptor of its dictionary
  //  -> file size of its dictionary
  //  -> mapped input of its dictionary
  std::map<int, int> dictionaries;
  std::map<int, off_t> dictionaryFilesizes;
  // Note: char* buf can be cast to appropriate struct
  // Struct will probably look like { implicit_oid, len, char* }
  std::map<int, char *> dictionariesBuf;

  /**
   * Code-generation-related
   */
  // Used to store memory positions of offset, buf and filesize in the generated
  // code
  std::map<string, llvm::AllocaInst *> NamedValuesBinaryCol;
  ParallelContext *const context;

  const char *posVar;      // = "offset";
  const char *bufVar;      // = "fileBuffer";
  const char *itemCtrVar;  // = "itemCtr";

  // Used to generate code
  void skipLLVM(RecordAttribute attName, llvm::Value *offset);
  void prepareArray(RecordAttribute attName);

  void nextEntry();

  void readAsLLVM(RecordAttribute attName,
                  std::map<RecordAttribute, ProteusValueMemory> &variables);

  /* Operate over char* */
  void readAsInt64LLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);

  /* Operates over int* */
  void readAsIntLLVM(RecordAttribute attName,
                     std::map<RecordAttribute, ProteusValueMemory> &variables);
  /* Operates over float* */
  void readAsFloatLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  /* Operates over bool* */
  void readAsBooleanLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  /* Not (fully) supported yet. Dictionary-based */
  void readAsStringLLVM(
      RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);

  // Generates a for loop that performs the file scan
  void scan(const Operator &producer);
};

#endif /* SCAN_TO_BLOCKS_SM_PLUGIN_HPP_ */
