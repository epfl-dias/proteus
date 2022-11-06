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

#ifndef PLUGINS_LLVM_HPP_
#define PLUGINS_LLVM_HPP_

#include <olap/util/cache-info.hpp>
#include <olap/util/context.hpp>
#include <olap/util/parallel-context.hpp>
#include <olap/values/expressionTypes.hpp>
#include <platform/common/common.hpp>
#include <platform/common/unsupported-operation.hpp>

/* Leads to incomplete type */
class OperatorState;
class Operator;
// class ProteusValueMemory;
// class ProteusValue;

// Used by all plugins
static const string activeLoop = "activeTuple";

/**
 * In principle, every readPath() method should deal with a record.
 * For some formats/plugins, however, (e.g. CSV) projection pushdown makes
 * significant difference in performance.
 */
typedef struct Bindings {
  const OperatorState *state;
  const ProteusValue record;
} Bindings;

enum PluginType { PGCSV, PGJSON, PGBINARY };
/**
 * @class Plugin abstract base class
 *
 * @see Karpathiotakis et al, VLDB2016
 */
class Plugin {
 public:
  virtual ~Plugin() { LOG(INFO) << "[PLUGIN: ] Collapsing plug-in"; }
  virtual string &getName() = 0;
  virtual void init() = 0;
  virtual void finish() = 0;
  virtual void generate(const Operator &producer, ParallelContext *context) = 0;
  /**
   * @param activeRelation Which relation's activeTuple is to be processed.
   *                          Does not have to be a native one
   *                          Relevant example:
   *                          for ( e <- employees, w <- employees.children )
   * yield ... Active tuple at some point will be the one of
   * "employees.children"
   */
  virtual ProteusValueMemory readPath(string activeRelation,
                                      Bindings wrappedBindings,
                                      const char *pathVar, RecordAttribute attr,
                                      ParallelContext *context) = 0;
  virtual ProteusValueMemory readValue(ProteusValueMemory mem_value,
                                       const ExpressionType *type,
                                       ParallelContext *context) = 0;
  virtual ProteusValue readCachedValue(CacheInfo info,
                                       const OperatorState &currState,
                                       ParallelContext *context) = 0;

  // Relevant for hashing visitors
  virtual ProteusValue hashValue(ProteusValueMemory mem_value,
                                 const ExpressionType *type,
                                 Context *context) = 0;
  /* Hash without re-interpreting. Mostly relevant if some value is cached
   * Meant for primitive values! */
  virtual ProteusValue hashValueEager(ProteusValue value,
                                      const ExpressionType *type,
                                      Context *context) = 0;

  /**
   * Not entirely sure which is the correct granularity for 'stuff to flush'
   * here.
   * XXX Atm we only have one (JSON) serializer, later on it'll have to be an
   * argument
   * XXX We probably also need sth for more isolated values
   */
  /**
   * We need an intermediate internal representation to deserialize to
   * before serializing back.
   *
   * Example: How would one convert from CSV to JSON?
   * Can't just crop CSV entries and flush as JSON
   *
   * 'Eager' variation: Caching related. Separates eager from lazy plugins
   * (i.e., all vs. JSON atm)
   * XXX 'Eager' flushing not tested
   */
  virtual void flushTuple(ProteusValueMemory mem_value,
                          llvm::Value *fileName) = 0;

 protected:
  virtual void flushValue(ProteusValueMemory mem_value,
                          const ExpressionType *type,
                          llvm::Value *fileName) = 0;
  virtual void flushValueEager(ProteusValue value, const ExpressionType *type,
                               llvm::Value *fileName) = 0;

 public:
  virtual void flushValue(Context *context, ProteusValueMemory mem_value,
                          const ExpressionType *type, std::string fileName) {
    flushValue(mem_value, type, context->CreateGlobalString(fileName.c_str()));
  }

  virtual void flushValueEager(Context *context, ProteusValue value,
                               const ExpressionType *type,
                               std::string fileName) {
    flushValueEager(value, type, context->CreateGlobalString(fileName.c_str()));
  }

  virtual void updateValue(ParallelContext *context, ProteusValueMemory mem_rid,
                           ProteusValueMemory mem_value,
                           const ExpressionType *type,
                           const std::string &fileName) {
    throw proteus::unsupported_operation("update");
  }

  virtual void updateValueEager(ParallelContext *context, ProteusValue rid,
                                ProteusValue value, const ExpressionType *type,
                                const std::string &fileName) {
    throw proteus::unsupported_operation("update eager");
  }

  /**
   * Relevant for collections' unnesting
   */
  virtual ProteusValueMemory initCollectionUnnest(
      ProteusValue val_parentObject) = 0;
  virtual ProteusValue collectionHasNext(
      ProteusValue val_parentObject, ProteusValueMemory mem_currentChild) = 0;
  virtual ProteusValueMemory collectionGetNext(
      ProteusValueMemory mem_currentChild) = 0;

  virtual void forEachInCollection(
      ParallelContext *context, ProteusValue val_parentObject,
      ProteusBareValue offset, ProteusBareValue step,
      ProteusBareValue val_parentObjectSize /* soon to be removed, the plugin
                                           should know the collection size */
      ,
      const std::function<void(ProteusValueMemory mem_currentChild,
                               llvm::MDNode *LoopID)> &) {
    throw proteus::unsupported_operation("for each in collection");
  }

  /**
   * Relevant when needed to materialize EAGERLY.
   * Otherwise, allocated type info suffices
   *
   * (i.e., not used that often)
   */
  virtual llvm::Value *getValueSize(ProteusValueMemory mem_value,
                                    const ExpressionType *type,
                                    ParallelContext *context) = 0;

  //    virtual typeID getOIDSize() = 0;
  virtual ExpressionType *getOIDType() = 0;
  virtual PluginType getPluginType() = 0;
  [[nodiscard]] virtual RecordType getRowType() const = 0;
  [[nodiscard]] virtual bool isPacked() const { return false; }

 protected:
  virtual void flushBeginList(llvm::Value *fileName) = 0;
  virtual void flushBeginBag(llvm::Value *fileName) = 0;
  virtual void flushBeginSet(llvm::Value *fileName) = 0;

  virtual void flushEndList(llvm::Value *fileName) = 0;
  virtual void flushEndBag(llvm::Value *fileName) = 0;
  virtual void flushEndSet(llvm::Value *fileName) = 0;

  virtual void flushDelim(llvm::Value *fileName, int depth = 0) = 0;
  virtual void flushDelim(llvm::Value *resultCtr, llvm::Value *fileName,
                          int depth = 0) = 0;

  virtual void flushOutput(llvm::Value *fileName) = 0;

 public:
  virtual void flushBeginList(Context *context, std::string fileName) {
    flushBeginList(context->CreateGlobalString(fileName.c_str()));
  }

  virtual void flushBeginBag(Context *context, std::string fileName) {
    flushBeginBag(context->CreateGlobalString(fileName.c_str()));
  }

  virtual void flushBeginSet(Context *context, std::string fileName) {
    flushBeginSet(context->CreateGlobalString(fileName.c_str()));
  }

  virtual void flushEndList(Context *context, std::string fileName) {
    flushEndList(context->CreateGlobalString(fileName.c_str()));
  }

  virtual void flushEndBag(Context *context, std::string fileName) {
    flushEndBag(context->CreateGlobalString(fileName.c_str()));
  }

  virtual void flushEndSet(Context *context, std::string fileName) {
    flushEndSet(context->CreateGlobalString(fileName.c_str()));
  }

  virtual void flushDelim(Context *context, std::string fileName,
                          int depth = 0) {
    flushDelim(context->CreateGlobalString(fileName.c_str()), depth);
  }

  virtual void flushDelim(Context *context, llvm::Value *resultCtr,
                          std::string fileName, int depth = 0) {
    flushDelim(resultCtr, context->CreateGlobalString(fileName.c_str()), depth);
  }

  virtual void flushOutput(Context *context, std::string fileName,
                           const ExpressionType *type = nullptr) {
    flushOutput(context->CreateGlobalString(fileName.c_str()));
  }

  virtual bool isLazy() { return false; }
};
#endif /* PLUGINS_LLVM_HPP_ */
