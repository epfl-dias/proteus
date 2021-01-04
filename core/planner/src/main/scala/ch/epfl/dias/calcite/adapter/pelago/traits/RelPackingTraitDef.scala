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
  PelagoPack,
  PelagoRel,
  PelagoUnpack
}
import org.apache.calcite.plan.{RelOptPlanner, RelTraitDef}
import org.apache.calcite.rel.RelNode

object RelPackingTraitDef { val INSTANCE = new RelPackingTraitDef }

/**
  * Definition of the device type trait.
  *
  * Target device type is a physical property (i.e. a trait) because it can be
  * changed without loss of information. The converter to do this is the
  * [[PelagoPack]] and [[PelagoUnpack]] operator.
  */
class RelPackingTraitDef protected () extends RelTraitDef[RelPacking] {
  override def getTraitClass: Class[RelPacking] = classOf[RelPacking]

  override def getSimpleName = "device"

  override def convert(
      planner: RelOptPlanner,
      rel: RelNode,
      to_packing: RelPacking,
      allowInfiniteCostConverters: Boolean
  ): RelNode = {
    if (rel.getConvention ne PelagoRel.CONVENTION) return null
    if (rel.getTraitSet.containsIfApplicable(to_packing)) return rel
    var p: RelNode = null
    if (to_packing eq RelPacking.Packed) p = PelagoPack.create(rel, to_packing)
    else p = PelagoUnpack.create(rel, to_packing)
    var newRel = planner.register(p, rel)
    val newTraitSet = rel.getTraitSet.replace(to_packing)
    if (!(newRel.getTraitSet == newTraitSet))
      newRel = planner.changeTraits(newRel, newTraitSet)
    newRel
  }
  override def canConvert(
      planner: RelOptPlanner,
      fromTrait: RelPacking,
      toTrait: RelPacking
  ) = true //fromTrait != toTrait;
  override def getDefault: RelPacking = RelPacking.UnPckd
}
