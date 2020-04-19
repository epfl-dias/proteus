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

#ifndef PROTEUS_RADIX_JOIN_BUILD_HPP
#define PROTEUS_RADIX_JOIN_BUILD_HPP
#include "expressions/expressions-generator.hpp"
#include "operators/operators.hpp"
#include "operators/scan.hpp"
#include "plugins/binary-internal-plugin.hpp"
#include "util/caching.hpp"
#include "util/functions.hpp"
#include "util/parallel-context.hpp"
#include "util/radix/joins/radix-join.hpp"

//#define DEBUGRADIX

/* valuePtr is relative to the payloadBuffer! */
typedef struct htEntry {
  int key;
  size_t valuePtr;
} htEntry;

struct relationBuf {
  /* Mem layout:
   * Consecutive <payload> chunks - payload type defined at runtime
   */
  // AllocaInst *mem_relation;
  // /* Size in bytes */
  // AllocaInst *mem_tuplesNo;
  // /* Size in bytes */
  // AllocaInst *mem_size;
  // /* (Current) Offset in bytes */
  // AllocaInst *mem_offset;
  // AllocaInst *mem_cachedTuplesNo;
  StateVar mem_relation_id;
  /* Size in bytes */
  StateVar mem_tuplesNo_id;
  /* Size in bytes */
  StateVar mem_size_id;
  /* (Current) Offset in bytes */
  StateVar mem_offset_id;
  StateVar mem_cachedTuplesNo_id;
};

struct kvBuf {
  // /* Mem layout:
  //  * Pairs of (int32 key, size_t payloadPtr)
  //  */
  // AllocaInst *mem_kv;
  // /* Size in bytes */
  // AllocaInst *mem_tuplesNo;
  // /* Size in bytes */
  // AllocaInst *mem_size;
  // /* (Current) Offset in bytes */
  // AllocaInst *mem_offset;
  /* Mem layout:
   * Pairs of (int32 key, size_t payloadPtr)
   */
  StateVar mem_kv_id;
  /* Size in bytes */
  StateVar mem_tuplesNo_id;
  /* Size in bytes */
  StateVar mem_size_id;
  /* (Current) Offset in bytes */
  StateVar mem_offset_id;
};

class RadixJoinBuild : public UnaryOperator {
 public:
  RadixJoinBuild(expression_t keyExpr, Operator *child,
                 ParallelContext *const context, string opLabel,
                 Materializer &mat, llvm::StructType *htEntryType,
                 size_t size
#ifdef LOCAL_EXEC
                 = 15000,
#else
                 = 30000000000,
#endif
                 size_t kvSize
#ifdef LOCAL_EXEC
                 = 15000,
#else
                 = 30000000000,
#endif
                 bool is_agg = false);
  virtual ~RadixJoinBuild();
  virtual void produce_(ParallelContext *context);
  //  void produceNoCache() ;
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual void consume(ParallelContext *const context,
                       const OperatorState &childState);
  Materializer &getMaterializer() { return mat; }
  virtual bool isFiltering() const { return true; }
  virtual RecordType getRowType() const {
    // FIXME: implement
    throw runtime_error("unimplemented RadixJoin:: getRowType()");
  }

  virtual int32_t *getClusterCounts(Pipeline *pip);
  virtual void registerClusterCounts(Pipeline *pip, int32_t *cnts);
  virtual void *getHTMemKV(Pipeline *pip);
  virtual void registerHTMemKV(Pipeline *pip, void *mem_kv);
  virtual void *getRelationMem(Pipeline *pip);
  virtual void registerRelationMem(Pipeline *pip, void *rel_mem);

  virtual llvm::StructType *getPayloadType() { return payloadType; }

 private:
  //     OperatorState* generate(Operator* op, OperatorState* childState);

  //     void runRadix() const;
  //     Value *radix_cluster_nopadding(struct relationBuf rel, struct kvBuf ht)
  //     const;
  llvm::Value *radix_cluster_nopadding(llvm::Value *mem_tuplesNo,
                                       llvm::Value *mem_kv_id) const;
  void initializeState();

  // //  char** findSideInCache(Materializer &mat) const;
  //     Scan* findSideInCache(Materializer &mat, bool isLeft) const;
  //     void placeInCache(Materializer &mat, bool isLeft) const;
  //     void updateRelationPointers() const;
  //     void freeArenas() const;
  relationBuf rel;
  kvBuf ht;

  size_t size;
  size_t kvSize;

  bool cached;
  bool is_agg;

  string htLabel;
  ParallelContext *const context;
  expression_t keyExpr;

  OutputPlugin *pg;

  Materializer &mat;
  llvm::StructType *htEntryType;
  llvm::StructType *payloadType;

  int32_t *clusterCounts[128];
  void *ht_mem_kv[128];
  void *relation_mem[128];

  // std::map<int32_t, int32_t *>    clusterCounts   ;
  // std::map<int32_t, void    *>    ht_mem_kv       ;
  // std::map<int32_t, void    *>    relation_mem    ;
};

#endif /* PROTEUS_RADIX_JOIN_BUILD_HPP */