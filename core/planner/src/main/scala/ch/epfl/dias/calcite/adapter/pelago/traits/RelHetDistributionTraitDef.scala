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

import ch.epfl.dias.calcite.adapter.pelago.rel.{
  PelagoRel,
  PelagoSplit,
  PelagoUnion
}
import org.apache.calcite.plan.{RelOptPlanner, RelTraitDef}
import org.apache.calcite.rel.RelNode

import java.util

object RelHetDistributionTraitDef {
  val INSTANCE = new RelHetDistributionTraitDef
}
class RelHetDistributionTraitDef protected ()
    extends RelTraitDef[RelHetDistribution] {
  override def getTraitClass: Class[RelHetDistribution] =
    classOf[RelHetDistribution]
  override def getSimpleName = "het_distribution"
  override def convert(
      planner: RelOptPlanner,
      rel: RelNode,
      distribution: RelHetDistribution,
      allowInfiniteCostConverters: Boolean
  ): RelNode = {
    if (rel.getConvention ne PelagoRel.CONVENTION) return null
    val inptraitSet = rel.getTraitSet
      .replace(RelDeviceType.X86_64)
      .replace(RelHomDistribution.SINGLE)
    val traitSet = rel.getTraitSet.replace(distribution)
    val input = rel
    if (!(rel.getTraitSet == inptraitSet)) { //      input = planner.register(planner.changeTraits(rel, inptraitSet), rel);
      return null
    }
    var router: RelNode = null
    if (
      (distribution eq RelHetDistribution.SPLIT) || (distribution eq RelHetDistribution.SPLIT_BRDCST)
    ) {
      if (!input.getTraitSet.contains(RelHetDistribution.SINGLETON)) return null
      if (!input.getTraitSet.containsIfApplicable(RelPacking.Packed))
        return null
//      if (!input.getTraitSet().containsIfApplicable(RelSplitPoint.NONE())) return null;
      router = PelagoSplit.create(input, distribution)
    } else { //      if (input.getTraitSet().containsIfApplicable(RelSplitPoint.NONE())) return null;
      if (!input.getTraitSet.contains(RelHetDistribution.SPLIT)) return null
      val c = input.getTraitSet
        .replace(RelComputeDevice.X86_64)
        .replace(RelDeviceType.X86_64)
      val g = input.getTraitSet
        .replace(RelComputeDevice.NVPTX)
        .replace(RelDeviceType.X86_64)
      var ing = input
      var inc = input
      planner.register(inc, rel)
      planner.register(ing, rel)
      if (!(ing.getTraitSet == g)) ing = planner.changeTraits(ing, g)
      if (!(inc.getTraitSet == c)) inc = planner.changeTraits(inc, c)
      planner.register(inc, rel)
      planner.register(ing, rel)
      router = PelagoUnion.create(util.List.of(inc, ing), all = true)
    }
    var newRel = planner.register(router, rel)
    if (!(newRel.getTraitSet == traitSet))
      newRel = planner.changeTraits(newRel, traitSet)
    newRel
  }
  override def canConvert(
      planner: RelOptPlanner,
      fromTrait: RelHetDistribution,
      toTrait: RelHetDistribution
  ): Boolean = {
    if (RelHetDistribution.SINGLETON eq fromTrait)
      return (toTrait eq RelHetDistribution.SPLIT) || (toTrait eq RelHetDistribution.SPLIT_BRDCST)
    else if (RelHetDistribution.SPLIT eq fromTrait)
      return toTrait eq RelHetDistribution.SINGLETON //FIXME: SPLIT can become SPLIT_BRDCST but I am not sure the executor supports it
    false // Can't convert SPLIT_BRDCST

  }
  override def getDefault: RelHetDistribution = RelHetDistribution.SINGLETON
}
