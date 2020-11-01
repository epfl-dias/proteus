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

#ifndef _NEST_HPP_
#define _NEST_HPP_

#include "lib/expressions/expressions-generator.hpp"
#include "lib/expressions/expressions-hasher.hpp"
#include "lib/expressions/path.hpp"
#include "lib/plugins/binary-internal-plugin.hpp"
#include "olap/expressions/expressions.hpp"
#include "olap/operators/monoids.hpp"
#include "operators.hpp"

#define DEBUGNEST

/**
 * Indicative query where a nest (..and an outer join) occur:
 * for (d <- Departments) yield set (D := d, E := for ( e <- Employees, e.dno =
 * d.dno) yield set e)
 *
 * TODO ADD MATERIALIZER / OUTPUT PLUGIN FOR NEST OPERATOR (?)
 * TODO Doesn't NEST require aliases for the two record arguments that are its
 * results?
 * XXX  Hashing keys is not enough - also need to compare the actual keys
 */
class Nest : public experimental::UnaryOperator {
 public:
  Nest(Monoid acc, expressions::Expression *outputExpr,
       expressions::Expression *pred,
       const list<expressions::InputArgument> &f_grouping,
       const list<expressions::InputArgument> &g_nullToZero, Operator *child,
       char *opLabel, Materializer &mat);
  ~Nest() override { LOG(INFO) << "Collapsing Nest operator"; }
  void produce_(ParallelContext *context) override;
  void consume(ParallelContext *context,
               const OperatorState &childState) override;
  Materializer &getMaterializer() { return mat; }
  [[nodiscard]] bool isFiltering() const override { return true; }

 private:
  void generateInsert(Context *context, const OperatorState &childState);
  /**
   * Once HT has been fully materialized, it is time to resume execution.
   * Note: generateProbe (should) not require any info reg. the previous op that
   * was called. Any info needed is (should be) in the HT that will now be
   * probed.
   */
  void generateProbe(ParallelContext *context) const;
  void generateSum(ParallelContext *context, const OperatorState &state,
                   llvm::AllocaInst *mem_accumulating) const;
  /**
   * We need a new accumulator for every resulting bucket of the HT
   */
  llvm::AllocaInst *resetAccumulator(ParallelContext *context) const;

  Monoid acc;
  expressions::Expression *outputExpr;
  expressions::Expression *pred;
  expressions::Expression *f_grouping;
  const list<expressions::InputArgument> &__attribute__((unused)) g_nullToZero;

  // Check TODO on naming above
  std::string aggregateName;

  char *htName;
  Materializer &mat;
};

#endif /* NEST_HPP_ */
