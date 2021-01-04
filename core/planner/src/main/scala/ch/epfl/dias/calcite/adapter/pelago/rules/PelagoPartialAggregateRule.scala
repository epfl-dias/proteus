/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
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

package ch.epfl.dias.calcite.adapter.pelago.rules

import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoAggregate
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHomDistribution
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptRule.{any, convert, operand}
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall}
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlSplittableAggFunction
import org.apache.calcite.util.ImmutableBitSet

import java.util
import scala.collection.JavaConverters._

object PelagoPartialAggregateRule {
  val INSTANCE = new PelagoPartialAggregateRule
}
class PelagoPartialAggregateRule protected ()
    extends RelOptRule(
      operand(classOf[PelagoAggregate], any),
      "PelagoPartialAggregate"
    ) {
  override def matches(call: RelOptRuleCall): Boolean = {
    val rel: PelagoAggregate = call.rel(0)
    rel.getTraitSet.containsIfApplicable(
      RelHomDistribution.SINGLE
    ) && //        (rel.getGroupCount() == 0) &&
    (rel.isGlobalAgg && !rel.isSplitted)
  }
  override def onMatch(call: RelOptRuleCall): Unit = {
    val rel: PelagoAggregate = call.rel(0)
    val aggCalls = new util.ArrayList[AggregateCall]
    val rexBuilder = rel.getCluster.getRexBuilder
    var i = rel.getGroupCount

    for (a <- rel.getAggCallList.asScala) {
      val s = a.getAggregation.unwrap(classOf[SqlSplittableAggFunction])
      if (s == null) return
      val list =
        new util.ArrayList[RexNode](rexBuilder.identityProjects(rel.getRowType))
      val reg: SqlSplittableAggFunction.Registry[RexNode] = (e: RexNode) => {
        def foo(e: RexNode) = {
          var i1 = list.indexOf(e)
          if (i1 < 0) {
            i1 = list.size
            list.add(e)
          }
          i1
        }
        foo(e)
      }
      val aTop =
        s.topSplit(rexBuilder, reg, rel.getGroupCount, rel.getRowType, a, i, -1)
//      if (a.getAggregation().getKind() == SqlKind.COUNT){
//        aggCalls.add(new AggregateCall(SqlSplittableAggFunction.CountSplitter.INSTANCE, AggFunction.SUM, a.isDistinct(), List.of(i), a.getType(), a.name));
//      } else {
//        aggCalls.add(new AggregateCall(a.getAggregation(), a.isDistinct(), List.of(i), a.getType(), a.name));
//      }
      aggCalls.add(aTop)
      i = i + 1
    }
    val topGroupSet = ImmutableBitSet.builder.set(0, rel.getGroupCount).build
    val locagg =
      rel.copy(
        convert(rel.getInput, RelHomDistribution.RANDOM),
        global = false,
        isSplitted = true
      )
    call.transformTo(
      call.getPlanner.register(
        PelagoAggregate.create(
          convert(locagg, RelHomDistribution.SINGLE),
          rel.getHints,
          topGroupSet,
          ImmutableList.of(topGroupSet),
          aggCalls,
          isGlobalAgg = true,
          isSplitted = true
        ),
        rel
      )
    )
  }
}
