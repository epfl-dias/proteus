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

#ifndef _NEST_OPT_HPP_
#define _NEST_OPT_HPP_

#include "expressions/expressions-generator.hpp"
#include "expressions/expressions-hasher.hpp"
#include "expressions/expressions.hpp"
#include "expressions/path.hpp"
#include "operators/monoids.hpp"
#include "operators/operators.hpp"
#include "plugins/binary-internal-plugin.hpp"

/**
 * Indicative query where a nest (..and an outer join) occur:
 * for (d <- Departments) yield set (D := d, E := for ( e <- Employees, e.dno =
 * d.dno) yield set e)
 *
 * NEST requires aliases for the record arguments that are its results.
 *
 * TODO ADD MATERIALIZER / OUTPUT PLUGIN FOR NEST OPERATOR (?)
 * XXX  Hashing keys is not enough - also need to compare the actual keys
 */
namespace opt {

class Nest : public UnaryOperator {
 public:
  Nest(vector<Monoid> accs, vector<expression_t> outputExprs,
       vector<string> aggrLabels, expression_t pred,
       const list<expressions::InputArgument> &f_grouping,
       const list<expressions::InputArgument> &g_nullToZero,
       Operator *const child, char *opLabel, Materializer &mat);
  virtual ~Nest() { LOG(INFO) << "Collapsing Nest operator"; }
  virtual void produce();
  virtual void consume(Context *const context, const OperatorState &childState);
  Materializer &getMaterializer() { return mat; }
  virtual bool isFiltering() const { return true; }

 private:
  void generateInsert(Context *context, const OperatorState &childState);
  /**
   * Once HT has been fully materialized, it is time to resume execution.
   * Note: generateProbe (should) not require any info reg. the previous op that
   * was called. Any info needed is (should be) in the HT that will now be
   * probed.
   */
  void generateProbe(Context *const context) const;
  void generateSum(expression_t outputExpr, Context *const context,
                   const OperatorState &state,
                   llvm::AllocaInst *mem_accumulating) const;
  void generateMul(expression_t outputExpr, Context *const context,
                   const OperatorState &state,
                   llvm::AllocaInst *mem_accumulating) const;
  void generateMax(expression_t outputExpr, Context *const context,
                   const OperatorState &state,
                   llvm::AllocaInst *mem_accumulating) const;
  void generateOr(expression_t outputExpr, Context *const context,
                  const OperatorState &state,
                  llvm::AllocaInst *mem_accumulating) const;
  void generateAnd(expression_t outputExpr, Context *const context,
                   const OperatorState &state,
                   llvm::AllocaInst *mem_accumulating) const;
  /**
   * We need a new accumulator for every resulting bucket of the HT
   */
  llvm::AllocaInst *resetAccumulator(expression_t outputExpr, Monoid acc) const;

  vector<Monoid> accs;
  vector<expression_t> outputExprs;
  expression_t pred;
  expression_t f_grouping;
  const list<expressions::InputArgument> &__attribute__((unused)) g_nullToZero;

  vector<string> aggregateLabels;

  char *htName;
  Materializer &mat;

  Context *context;
};

}  // namespace opt
#endif /* NEST_OPT_HPP_ */
