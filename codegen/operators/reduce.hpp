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

#ifndef REDUCE_HPP_
#define REDUCE_HPP_

#include "expressions/expressions-flusher.hpp"
#include "expressions/expressions-generator.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"

//#ifdef DEBUG
#define DEBUGREDUCE
//#endif

/**
 * TODO ADD MATERIALIZER / OUTPUT PLUGIN FOR REDUCE OPERATOR
 * ADD 'SERIALIZER' SUPPORT IN ALL CASES, NOT ONLY LIST/BAG AS NOW
 */
class Reduce : public UnaryOperator {
 public:
  Reduce(Monoid acc, expressions::Expression *outputExpr,
         expressions::Expression *pred, Operator *const child,
         Context *context);
  virtual ~Reduce() { LOG(INFO) << "Collapsing Reduce operator"; }
  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  virtual bool isFiltering() const { return true; }

 private:
  Context *__attribute__((unused)) context;

  Monoid acc;
  expressions::Expression *outputExpr;
  expressions::Expression *pred;
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

#endif /* REDUCE_HPP_ */
