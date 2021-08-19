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

#ifndef _NEST_RADIX_HPP_
#define _NEST_RADIX_HPP_

#include <olap/expressions/expressions.hpp>
#include <olap/operators/monoids.hpp>
#include <olap/util/parallel-context.hpp>
#include <platform/util/radix/aggregations/radix-aggr.hpp>

#include "lib/expressions/expressions-dot-evaluator.hpp"
#include "lib/expressions/expressions-generator.hpp"
#include "lib/expressions/expressions-hasher.hpp"
#include "lib/expressions/path.hpp"
#include "operators.hpp"
#include "radix-join.hpp"

#define DEBUGRADIX_NEST
/**
 * Indicative query where a nest (..and an outer join) occur:
 * for (d <- Departments) yield set (D := d, E := for ( e <- Employees, e.dno =
 * d.dno) yield set e)
 *
 * NEST requires aliases for the record arguments that are its results.
 *
 * TODO ADD MATERIALIZER / OUTPUT PLUGIN FOR NEST OPERATOR (?)
 * XXX  Hashing keys is not enough - also need to compare the actual keys
 *
 * TODO Different types of (input) collections enforce different dot equality
 * requirements!! Example: OID & 'CollectionID' (i.e., pgID) must participate in
 * dot equality for elems for lists!!
 */
namespace radix {

/* valuePtr is relative to the payloadBuffer! */
typedef struct htEntry {
  size_t key;
  size_t valuePtr;
} htEntry;

struct relationBuf {
  /* Mem layout:
   * Consecutive <payload> chunks - payload type defined at runtime
   */
  llvm::AllocaInst *mem_relation;
  /* Size in bytes */
  llvm::AllocaInst *mem_tuplesNo;
  /* Size in bytes */
  llvm::AllocaInst *mem_size;
  /* (Current) Offset in bytes */
  llvm::AllocaInst *mem_offset;
};

struct kvBuf {
  /* Mem layout:
   * Pairs of (size_t key, size_t payloadPtr)
   */
  llvm::AllocaInst *mem_kv;
  /* Size in bytes */
  llvm::AllocaInst *mem_tuplesNo;
  /* Size in bytes */
  llvm::AllocaInst *mem_size;
  /* (Current) Offset in bytes */
  llvm::AllocaInst *mem_offset;
};

class Nest : public UnaryOperator {
 public:
  Nest(Context *const context, std::vector<Monoid> accs,
       std::vector<expression_t> outputExprs, std::vector<string> aggrLabels,
       expression_t pred, std::vector<expression_t> f_grouping,
       expression_t g_nullToZero, Operator *const child,
       const std::string &opLabel, Materializer &mat);
  Nest(Context *const context, std::vector<Monoid> accs,
       std::vector<expression_t> outputExprs, std::vector<string> aggrLabels,
       expression_t pred, expression_t f_grouping, expression_t g_nullToZero,
       Operator *const child, const std::string &opLabel, Materializer &mat);
  ~Nest() override { LOG(INFO) << "Collapsing Nest operator"; }
  void produce_(ParallelContext *context) override;
  void consume(Context *const context,
               const OperatorState &childState) override;
  Materializer &getMaterializer() { return mat; }
  bool isFiltering() const override { return true; }

  RecordType getRowType() const override {
    std::vector<RecordAttribute *> attrs;
    for (const auto &attr : f_grouping_vec) {
      attrs.emplace_back(new RecordAttribute{attr.getRegisteredAs()});
    }
    size_t i = 0;
    for (const auto &attr : aggregateLabels) {
      attrs.emplace_back(new RecordAttribute{
          htName, attr, outputExprs[i++].getExpressionType()});
    }
    return attrs;
  }

 private:
  // void generateInsert(Context* context, const OperatorState& childState);
  /* Very similar to radix join building phase! */
  // void buildHT(Context* context, const OperatorState& childState);
  void probeHT() const;
  // Value* radix_cluster_nopadding(struct relationBuf rel, struct kvBuf ht)
  // const;
  /**
   * Once HT has been fully materialized, it is time to resume execution.
   * Note: generateProbe (should) not require any info reg. the previous op that
   * was called. Any info needed is (should be) in the HT that will now be
   * probed.
   */
  void generateProbe(Context *const context) const;

  std::map<RecordAttribute, ProteusValueMemory> reconstructResults(
      llvm::Value *htBuffer, llvm::Value *idx,
      const StateVar &relR_mem_relation_id) const;
  /**
   * We need a new accumulator for every resulting bucket of the HT
   */
  llvm::AllocaInst *resetAccumulator(expression_t outputExpr, Monoid acc) const;
  // void updateRelationPointers() const;

  std::vector<Monoid> accs;
  std::vector<expression_t> outputExprs;
  expression_t pred;
  expression_t f_grouping;
  expression_t __attribute__((unused)) g_nullToZero;
  std::vector<expression_t> f_grouping_vec;

  std::vector<string> aggregateLabels;

  std::string htName;
  Materializer mat;
  ParallelContext *context;
  RadixJoinBuild *build;

  /**
   * Relevant to radix-based HT
   */
  /* What the keyType is */
  /* XXX int32 FOR NOW   */
  /* If it is not int32 to begin with, hash it to make it so */
  llvm::StructType *payloadType;
  llvm::Type *keyType;
  llvm::StructType *htEntryType;
  // struct relationBuf relR;
  // struct kvBuf htR;
  HT *HT_per_cluster;
  llvm::StructType *htClusterType;
  // Raw Buffers
  char *relationR;
  char **ptr_relationR;
  char *kvR;
};

}  // namespace radix
#endif /* NEST_OPT_HPP_ */
