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

#ifndef _RADIX_JOIN_HPP_
#define _RADIX_JOIN_HPP_

#include <operators/join/radix/radix-join-build.hpp>

#include "expressions/expressions-generator.hpp"
#include "operators/operators.hpp"
#include "operators/scan.hpp"
#include "plugins/binary-internal-plugin.hpp"
#include "util/caching.hpp"
#include "util/functions.hpp"
#include "util/parallel-context.hpp"
#include "util/radix/joins/radix-join.hpp"

class RadixJoin : public BinaryOperator {
 public:
  RadixJoin(const expressions::BinaryExpression &predicate, Operator *leftChild,
            Operator *rightChild, Context *const context, const char *opLabel,
            Materializer &matLeft, Materializer &matRight);
  virtual ~RadixJoin();
  virtual void produce_(ParallelContext *context);
  //    void produceNoCache() ;
  virtual void consume(Context *const context, const OperatorState &childState);
  Materializer &getMaterializerLeft() const {
    return buildR->getMaterializer();
  }
  Materializer &getMaterializerRight() const {
    return buildS->getMaterializer();
  }
  virtual bool isFiltering() const { return true; }

  virtual RecordType getRowType() const {
    std::vector<RecordAttribute *> ret;

    for (const auto &mexpr : buildR->getMaterializer().getWantedFields()) {
      ret.emplace_back(new RecordAttribute(*mexpr));
    }

    for (const auto &mexpr : buildS->getMaterializer().getWantedFields()) {
      ret.emplace_back(new RecordAttribute(*mexpr));
    }

    return ret;
  }

 private:
  void runRadix() const;
  // Value *radix_cluster_nopadding(size_t mem_tuplesNo_id, size_t mem_kv_id)
  // const;

  // void initializeStateOneSide(size_t       size,
  //                             relationBuf &rel,
  //                             size_t       kvSize,
  //                             kvBuf       &ht);

  //    char** findSideInCache(Materializer &mat) const;
  // Scan* findSideInCache(Materializer &mat, bool isLeft) const;
  // void placeInCache(Materializer &mat, bool isLeft) const;
  // void updateRelationPointers() const;
  // void freeArenas() const;
  // struct relationBuf relR;
  // struct relationBuf relS;

  /* What the keyType is */
  /* XXX int32 FOR NOW   */
  /* If it is not int32 to begin with, hash it to make it so */
  llvm::Type *keyType;
  llvm::StructType *htEntryType;

  RadixJoinBuild *buildR;
  RadixJoinBuild *buildS;

  // size_t htR_mem_kv_id;
  // size_t htS_mem_kv_id;

  // size_t relR_mem_relation_id;
  // size_t relS_mem_relation_id;

  // struct kvBuf htR;
  // struct kvBuf htS;

  // HT *HT_per_cluster;
  llvm::StructType *htClusterType;

  // char *relationR;
  // char *relationS;
  // char **ptr_relationR;
  // char **ptr_relationS;
  // char *kvR;
  // char *kvS;

  string htLabel;
  ParallelContext *const context;
  void *flush_fun;

  // /* Cache- related */
  // bool cachedLeft;
  // bool cachedRight;
};

#endif /* _RADIX_JOIN_HPP_ */
