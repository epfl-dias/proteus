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

package ch.epfl.dias.calcite.adapter.pelago.traits

import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitDef
import org.apache.calcite.rel.RelNode

object RelSplitPointTraitDef { val INSTANCE = new RelSplitPointTraitDef }
class RelSplitPointTraitDef protected () extends RelTraitDef[RelSplitPoint] {
  override def getTraitClass: Class[RelSplitPoint] = classOf[RelSplitPoint]
  override def getSimpleName = "split"
  override def convert(
      planner: RelOptPlanner,
      rel: RelNode,
      toDevice: RelSplitPoint,
      allowInfiniteCostConverters: Boolean
  ): RelNode = {
    if (!rel.getTraitSet.containsIfApplicable(toDevice)) return null
    rel
  }

  override def canConvert(
      planner: RelOptPlanner,
      fromTrait: RelSplitPoint,
      toDevice: RelSplitPoint
  ): Boolean =
    (fromTrait eq RelSplitPoint.NONE) || (toDevice eq RelSplitPoint.NONE)
  override def getDefault: RelSplitPoint = RelSplitPoint.NONE
}
