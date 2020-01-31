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

#include "plugins/plugins.hpp"
#include "storage/storage-manager.hpp"

class ParallelContext;

class BinaryBlockPlugin : public Plugin {
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
  BinaryBlockPlugin(ParallelContext *context, string fnamePrefix,
                    RecordType rec, std::vector<RecordAttribute *> &whichFields,
                    bool load);

 public:
  BinaryBlockPlugin(ParallelContext *context, string fnamePrefix,
                    RecordType rec,
                    std::vector<RecordAttribute *> &whichFields);

  BinaryBlockPlugin(ParallelContext *context, string fnamePrefix,
                    RecordType rec);

  ~BinaryBlockPlugin() override;
  string &getName() override { return fnamePrefix; }
  void init() override;

  void generate(const Operator &producer) override;
  void finish() override;
  ProteusValueMemory readPath(string activeRelation, Bindings bindings,
                              const char *pathVar,
                              RecordAttribute attr) override;
  ProteusValueMemory readValue(ProteusValueMemory mem_value,
                               const ExpressionType *type) override;
  ProteusValue readCachedValue(CacheInfo info,
                               const OperatorState &currState) override;

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
                            const ExpressionType *type) override;

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

 protected:
  void flushOutputInternal(Context *context, std::string fileName,
                           const ExpressionType *type);

  void flushValueInternal(Context *context, ProteusValueMemory mem_value,
                          const ExpressionType *type, std::string fileName);

  void flushOutput(llvm::Value *fileName) override {
    string error_msg = "Reached a deprecated function";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValue(ProteusValueMemory mem_value, const ExpressionType *type,
                  llvm::Value *fileName) override {
    string error_msg = "Reached a deprecated function";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void flushValueEager(ProteusValue value, const ExpressionType *type,
                       llvm::Value *fileName) override {
    string error_msg = "Reached a deprecated function";
    LOG(ERROR) << error_msg;
    throw runtime_error(error_msg);
  }

  void finalize_data();
  [[nodiscard]] RecordType getRowType() const override;

  [[nodiscard]] virtual llvm::Value *getSession() const { return nullptr; }

  virtual std::pair<llvm::Value *, llvm::Value *> getPartitionSizes(
      llvm::Value *) const;
  virtual void freePartitionSizes(llvm::Value *) const;

  virtual llvm::Value *getDataPointersForFile(size_t i, llvm::Value *) const;
  virtual void freeDataPointersForFile(size_t i, llvm::Value *) const;

  virtual void releaseSession(llvm::Value *) const {}

  bool isLazy() override { return true; }

  std::vector<RecordAttribute *> wantedFields;
  std::vector<std::vector<mem_file>> wantedFieldsFiles;

 private:
  // Schema info provided
  RecordType rec;
  llvm::Value *blockSize;

 protected:
  size_t Nparts;

 protected:
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

 protected:
  ParallelContext *const context;

 private:
  const char *posVar;      // = "offset";
  const char *bufVar;      // = "fileBuffer";
  const char *itemCtrVar;  // = "itemCtr";

  // Used to generate code
  void skipLLVM(RecordAttribute attName, llvm::Value *offset);

  void nextEntry();

  void readAsLLVM(RecordAttribute attName,
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

  ProteusValueMemory readProteusValue(ProteusValueMemory val,
                                      const ExpressionType *type);

  // Generates a for loop that performs the file scan
  void scan(const Operator &producer);
};

#endif /* BINARY_BLOCK_PLUGIN_HPP_ */
