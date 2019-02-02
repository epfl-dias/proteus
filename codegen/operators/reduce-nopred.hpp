/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Labaratory (DIAS)
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

#ifndef REDUCENOPRED_HPP_
#define REDUCENOPRED_HPP_

#include "expressions/expressions-flusher.hpp"
#include "expressions/expressions-generator.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"

//#ifdef DEBUG
#define DEBUGREDUCENOPRED
//#endif

/**
 * In many cases, the plan that is produced includes
 * a Reduce Operator with predicate = true.
 *
 * This simplified operator implementation does not perform
 * whether p is true
 */
class ReduceNoPred : public UnaryOperator {
 public:
  ReduceNoPred(Monoid acc, expressions::Expression *outputExpr,
               Operator *const child, Context *context);
  virtual ~ReduceNoPred() { LOG(INFO) << "Collapsing Reduce operator"; }
  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return getChild()->isFiltering(); }

 private:
  Context *__attribute__((unused)) context;

  Monoid acc;
  expressions::Expression *outputExpr;
  llvm::AllocaInst *mem_accumulating;

  void generate(Context *const context, const OperatorState &childState) const;
  void generateSum(Context *const context,
                   const OperatorState &childState) const;
  void generateMul(Context *const context,
                   const OperatorState &childState) const;
  void generateMax(Context *const context,
                   const OperatorState &childState) const;
  void generateAnd(Context *const context,
                   const OperatorState &childState) const;
  void generateOr(Context *const context,
                  const OperatorState &childState) const;
  void generateUnion(Context *const context,
                     const OperatorState &childState) const;
  void generateBagUnion(Context *const context,
                        const OperatorState &childState) const;
  void generateAppend(Context *const context,
                      const OperatorState &childState) const;

  void flushResult();
};

#endif /* REDUCENOPRED_HPP_ */
