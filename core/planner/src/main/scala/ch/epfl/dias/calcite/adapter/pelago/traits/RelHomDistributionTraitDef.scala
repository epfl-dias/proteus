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

import ch.epfl.dias.calcite.adapter.pelago.rel.{PelagoRel, PelagoRouter}
import org.apache.calcite.plan.{RelOptPlanner, RelTraitDef}
import org.apache.calcite.rel.RelNode

object RelHomDistributionTraitDef {
  val INSTANCE = new RelHomDistributionTraitDef
}

class RelHomDistributionTraitDef protected ()
    extends RelTraitDef[RelHomDistribution] {
  override def getTraitClass: Class[RelHomDistribution] =
    classOf[RelHomDistribution]
  override def getSimpleName = "hom_distribution"
  override def convert(
      planner: RelOptPlanner,
      rel: RelNode,
      distribution: RelHomDistribution,
      allowInfiniteCostConverters: Boolean
  ): RelNode = {
    if (rel.getTraitSet.containsIfApplicable(distribution)) return rel
    if (
      (rel.getConvention ne PelagoRel.CONVENTION) ||
      !rel.getTraitSet.containsIfApplicable(RelDeviceType.X86_64)
    ) {
      return null
    }

    val traitSet = rel.getTraitSet.replace(distribution)
    val input = rel
//    if (!rel.getTraitSet().equals(inptraitSet)) {
////      input = planner.register(planner.changeTraits(rel, inptraitSet), rel);
//      return null;
//    }
    val router = PelagoRouter.create(input, distribution)
    var newRel = planner.register(router, rel)
    if (!(newRel.getTraitSet == traitSet)) {
      newRel = planner.register(planner.changeTraits(newRel, traitSet), rel)
    }
    newRel
  }
  override def canConvert(
      planner: RelOptPlanner,
      fromTrait: RelHomDistribution,
      toTrait: RelHomDistribution
  ): Boolean = {
    if (RelHomDistribution.SINGLE eq fromTrait)
      return (toTrait eq RelHomDistribution.RANDOM) || (toTrait eq RelHomDistribution.BRDCST)
    else if (RelHomDistribution.RANDOM eq fromTrait)
      return toTrait eq RelHomDistribution.SINGLE //FIXME: RANDOM can become BRDCST but I am not sure the executor supports it
    false // Can't convert BRDCST

  }
  override def getDefault: RelHomDistribution = RelHomDistribution.SINGLE
}
