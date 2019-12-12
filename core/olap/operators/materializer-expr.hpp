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

#ifndef _MATERIALIZER_EXPR_HPP_
#define _MATERIALIZER_EXPR_HPP_

#include "expressions/expressions-generator.hpp"
#include "operators/operators.hpp"
#include "util/caching.hpp"
#include "util/functions.hpp"

struct matBuf {
  /* Mem layout:
   * Consecutive <payload> chunks - payload type defined at runtime
   */
  llvm::AllocaInst *mem_buffer;
  /* Size in bytes */
  llvm::AllocaInst *mem_tuplesNo;
  /* Size in bytes */
  llvm::AllocaInst *mem_size;
  /* (Current) Offset in bytes */
  llvm::AllocaInst *mem_offset;
};

/**
 * Issue when attempting realloc() on server
 */
class ExprMaterializer : public UnaryOperator {
 public:
  ExprMaterializer(expressions::Expression *expr, Operator *const child,
                   Context *const context, char *opLabel);
  ExprMaterializer(expressions::Expression *expr, int linehint,
                   Operator *const child, Context *const context,
                   char *opLabel);
  virtual ~ExprMaterializer();
  virtual void produce_(ParallelContext *context);
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return false; }

 private:
  void freeArenas() const;
  void updateRelationPointers() const;

  llvm::StructType *toMatType;
  struct matBuf opBuffer;
  char *rawBuffer;
  char **ptr_rawBuffer;

  Context *const context;
  expressions::Expression *toMat;
  string opLabel;
};

#endif
