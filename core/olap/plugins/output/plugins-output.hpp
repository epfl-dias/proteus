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

#ifndef PLUGINS_OUTPUT_HPP_
#define PLUGINS_OUTPUT_HPP_

#include <llvm/IR/DerivedTypes.h>

#include "expressions/expressions.hpp"
#include "plugins/plugins.hpp"
//#include "util/raw-catalog.hpp"

class Context;
// TODO Refactor into multiple materializers
/**
 * What to do with each attribute contained in the payload of a materializing
 * operation
 */
// Eager: Place attribute converted to its original datatype
// Lazy: Place pointer to attribute in the raw file
// Intermediate: Copy (unparsed) attribute and place the copy
// Binary: Convert to concise form before placing (current example: JSON to
// BSON)
enum materialization_mode { EAGER, LAZY, INTERMEDIATE, BINARY, CSV, JSON };

class Materializer {
  /**
   * Fields: All the attributes involved - oids and whatnot
   * Expressions: The expressions (record projections!) in which the previous
   * fields are used OIDs: The plugin-created identifiers
   *
   * FIXME There is DEFINITELY a better way to do this.
   */
 public:
  Materializer(std::vector<RecordAttribute *> whichFields,
               std::vector<expression_t> wantedExpressions,
               std::vector<RecordAttribute *> whichOIDs,
               std::vector<materialization_mode> outputMode_);

  /* FIXME TODO
   * Unfortunately, too many of the experiments rely on this constructor */
  Materializer(std::vector<RecordAttribute *> whichFields,
               std::vector<materialization_mode> outputMode_)
      __attribute__((deprecated));

  /*
   * XXX New constructor, hoping to simplify process
   * Materializes eagerly
   *
   * XXX ORDER OF expression fields matters!! OIDs need to be placed first!
   */
  Materializer(std::vector<expression_t> wantedExpressions);

  ~Materializer() {}

  const std::vector<RecordAttribute *> &getWantedFields() const {
    return wantedFields;
  }
  const std::vector<RecordAttribute *> &getWantedOIDs() {
    if (oidsProvided) {
      return wantedOIDs;
    } else {
      /* HACK to avoid crashes in deprecated test cases!
       * FIXME This is deprecated and should NOT be used.
       * For example, it still breaks 3way joins */
      std::vector<RecordAttribute *> *newOIDs =
          new std::vector<RecordAttribute *>();
      set<RecordAttribute>::iterator it = tupleIdentifiers.begin();
      for (; it != tupleIdentifiers.end(); it++) {
        RecordAttribute *attr = new RecordAttribute(
            it->getRelationName(), it->getAttrName(), it->getOriginalType());
        newOIDs->push_back(attr);
      }
      return *newOIDs;  // wantedOIDs;
    }
  }
  const std::vector<expression_t> &getWantedExpressions() const {
    return wantedExpressions;
  }
  const std::vector<materialization_mode> &getOutputMode() const {
    return outputMode;
  }

  /* XXX Remove. OIDs are not sth to be specified
   * by checking bindings. It's the query plan that should
   * provide them */
  void addTupleIdentifier(RecordAttribute attr) {
    tupleIdentifiers.insert(attr);
  }
  //    const set<RecordAttribute>& getTupleIdentifiers() const {
  //        return tupleIdentifiers;
  //    }
 private:
  /**
   *  CONVENTIONS:
   *  wantedExpressions include activeTuple.
   *  wantedFields do not!
   */
  std::vector<expression_t> wantedExpressions;

  std::vector<RecordAttribute *> wantedFields;
  std::vector<RecordAttribute *> wantedOIDs;
  std::vector<materialization_mode> outputMode;
  // int tupleIdentifiers;
  std::set<RecordAttribute> tupleIdentifiers;

  bool oidsProvided;
};

class OutputPlugin {
 public:
  OutputPlugin(Context *const context, Materializer &materializer,
               const map<RecordAttribute, ProteusValueMemory> *bindings);
  ~OutputPlugin() {}

  llvm::StructType *getPayloadType() { return payloadType; }

  bool hasComplexTypes() { return isComplex; }

  /* static - not to be used with eager modes */
  int getPayloadTypeSize() { return payloadTypeSize; }

  /**
   * To be used when we consider eagerly materializing
   * collections, strings, etc.
   */
  llvm::Value *getRuntimePayloadTypeSize();

  void setBindings(const map<RecordAttribute, ProteusValueMemory> *bindings) {
    currentBindings = bindings;
  }
  const map<RecordAttribute, ProteusValueMemory> &getBindings() const {
    return *currentBindings;
  }
  std::vector<llvm::Type *> *getMaterializedTypes() {
    return materializedTypes;
  }
  llvm::Value *convert(llvm::Type *currType, llvm::Type *materializedType,
                       llvm::Value *val);

 private:
  // Schema info provided
  const Materializer &materializer;

  // Code-generation-related
  Context *const context;
  const map<RecordAttribute, ProteusValueMemory> *currentBindings;
  llvm::StructType *payloadType;
  std::vector<llvm::Type *> *materializedTypes;

  /* Report whether payload comprises only scalars (or not) */
  bool isComplex;
  /* Accumulated size of the various tuple identifiers */
  int identifiersTypeSize;
  /* Static computation of size in case of late materialization */
  /* Size per-binding, and total size */
  std::vector<int> fieldSizes;
  int payloadTypeSize;

  llvm::Type *chooseType(const ExpressionType *exprType, llvm::Type *currType,
                         materialization_mode mode);
};

#endif /* PLUGINS_OUTPUT_HPP_ */
