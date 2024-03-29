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

#ifndef BINARY_BLOCK_PLUGIN_HPP_
#define BINARY_BLOCK_PLUGIN_HPP_

#include <memory>
#include <storage/storage-manager.hpp>

#include "plugins.hpp"

class ParallelContext;

class BinaryBlockPlugin : public Plugin {
 public:
  static constexpr auto type = "block";

 protected:
  BinaryBlockPlugin(ParallelContext *context, const string &fnamePrefix,
                    RecordType rec,
                    const std::vector<RecordAttribute *> &whichFields,
                    bool load);

 public:
  BinaryBlockPlugin(ParallelContext *context, const string &fnamePrefix,
                    RecordType rec,
                    const std::vector<RecordAttribute *> &whichFields);

  string &getName() override { return fnamePrefix; }
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
                               ParallelContext *context) override;

  ProteusValue hashValue(ProteusValueMemory mem_value,
                         const ExpressionType *type, Context *context) override;
  ProteusValue hashValueEager(ProteusValue value, const ExpressionType *type,
                              Context *context) override;

  ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) override {
    string error_msg =
        "[BinaryBlockPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValue collectionHasNext(ProteusValue val_parentObject,
                                 ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryBlockPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }
  ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) override {
    string error_msg =
        "[BinaryBlockPlugin: ] Binary col. files do not contain collections";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void forEachInCollection(
      ParallelContext *context, ProteusValue val_parentObject,
      ProteusBareValue offset, ProteusBareValue step,
      ProteusBareValue val_parentObjectSize,
      const std::function<void(ProteusValueMemory mem_currentChild,
                               llvm::MDNode *LoopID)> &) override;

  void flushTuple(ProteusValueMemory mem_value,
                  llvm::Value *fileName) override {
    string error_msg = "[BinaryBlockPlugin: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(Context *context, ProteusValueMemory mem_value,
                  const ExpressionType *type, std::string fileName) override;

  void flushValueEager(Context *context, ProteusValue mem_value,
                       const ExpressionType *type,
                       std::string fileName) override;

  llvm::Value *getValueSize(ProteusValueMemory mem_value,
                            const ExpressionType *type,
                            ParallelContext *context) override;

  ExpressionType *getOIDType() override { return new Int64Type(); }

  PluginType getPluginType() override { return PGBINARY; }

  void flushBeginList(llvm::Value *fileName) override;

  void flushBeginBag(llvm::Value *fileName) override {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushBeginSet(llvm::Value *fileName) override {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndList(llvm::Value *fileName) override;

  void flushEndBag(llvm::Value *fileName) override {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushEndSet(llvm::Value *fileName) override {
    string error_msg = "[ScanToBlocksSM: ] Flush not implemented yet";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushDelim(llvm::Value *fileName, int depth) override;

  void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                  int depth) override;

  void flushOutput(Context *context, std::string fileName,
                   const ExpressionType *type) override;

  [[nodiscard]] RecordType getRowType() const override;

 protected:
  void loadData(ParallelContext *context, data_loc loc);

  void flushOutputInternal(Context *context, std::string fileName,
                           const ExpressionType *type);

  void flushValueInternal(Context *context, ProteusValueMemory mem_value,
                          const ExpressionType *type, std::string fileName);

  void flushOutput(llvm::Value *fileName) override {
    string error_msg = "Reached a deprecated function";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(ProteusValueMemory mem_value, const ExpressionType *,
                  llvm::Value *fileName) override {
    string error_msg = "Reached a deprecated function";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValueEager(ProteusValue value, const ExpressionType *,
                       llvm::Value *fileName) override {
    string error_msg = "Reached a deprecated function";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void finalize_data(ParallelContext *context);

  [[nodiscard]] virtual llvm::Value *getSession(ParallelContext *) const {
    return nullptr;
  }

  virtual std::pair<llvm::Value *, llvm::Value *> getPartitionSizes(
      ParallelContext *context, llvm::Value *) const;
  virtual void freePartitionSizes(ParallelContext *context,
                                  llvm::Value *) const;

  virtual llvm::Value *getDataPointersForFile(ParallelContext *context,
                                              size_t i, llvm::Value *) const;
  virtual void freeDataPointersForFile(ParallelContext *context, size_t i,
                                       llvm::Value *) const;

  virtual void releaseSession(ParallelContext *context, llvm::Value *) const {}

  bool isLazy() override { return true; }

  [[nodiscard]] bool isPacked() const override { return true; }

  std::vector<RecordAttribute *> wantedFields;

 protected:
  std::vector<FileRequest> wantedFieldsFiles;
  std::vector<size_t> fieldSizes;

 private:
  // Schema info provided
  RecordType rec;

 protected:
  size_t Nparts;

 protected:
  std::string fnamePrefix;

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

 private:
  static constexpr auto posVar = "offset";
  static constexpr auto bufVar = "buf";
  static constexpr auto itemCtrVar = "itemCtr";

  // Used to generate code
  void skipLLVM(ParallelContext *context, RecordAttribute attName,
                llvm::Value *offset);

  void nextEntry(ParallelContext *context, llvm::Value *blockSize);

  void readAsLLVM(ParallelContext *context, RecordAttribute attName,
                  std::map<RecordAttribute, ProteusValueMemory> &variables);

  /* Operates over int* */
  void readAsIntLLVM(ParallelContext *context, RecordAttribute attName,
                     std::map<RecordAttribute, ProteusValueMemory> &variables);
  /* Operates over float* */
  void readAsFloatLLVM(
      ParallelContext *context, RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);
  /* Operates over bool* */
  void readAsBooleanLLVM(
      ParallelContext *context, RecordAttribute attName,
      std::map<RecordAttribute, ProteusValueMemory> &variables);

  ProteusValueMemory readProteusValue(ProteusValueMemory val,
                                      const ExpressionType *type,
                                      ParallelContext *context);

  // Generates a for loop that performs the file scan
  void scan(const Operator &producer, ParallelContext *context);

 private:
  const void **getDataForField(size_t i);
  void freeDataForField(size_t i, const void **d);
  int64_t *getTuplesPerPartition();
  void freeTuplesPerPartition(int64_t *);

  /**
   * Handles relative/absolute paths.
   *
   * Relative paths are assumed to be in shared memory
   * Absolute paths are assumed to be regular files.
   * e.g:
   * For `relName` return relName/relName so attributes are written to
   * relName/relName.attrName
   * For `/my/absolute/path/relName` instead return
   * /my/absolute/path/relName/relname so attributes are written to
   * /my/absolute/path/relName/relname.attrName
   */
  std::string normalizeOutputPath(std::string);

  friend const void **getDataForField(size_t i, BinaryBlockPlugin *pg);
  friend void freeDataForField(size_t i, const void **d, BinaryBlockPlugin *pg);
  friend int64_t *getTuplesPerPartition(BinaryBlockPlugin *pg);
  friend void freeTuplesPerPartition(int64_t *, BinaryBlockPlugin *pg);

  void nextEntryInBlock(llvm::Value *mem_itemCtr, ParallelContext *context,
                        ProteusBareValue step);
};

#endif /* BINARY_BLOCK_PLUGIN_HPP_ */
