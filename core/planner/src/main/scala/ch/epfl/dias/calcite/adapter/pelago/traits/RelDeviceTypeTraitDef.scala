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

import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoDeviceCross
import org.apache.calcite.plan.{RelOptPlanner, RelTraitDef}
import org.apache.calcite.rel.RelNode

object RelDeviceTypeTraitDef { val INSTANCE = new RelDeviceTypeTraitDef }

/**
  * Definition of the device type trait.
  *
  * <p>Target device type is a physical property (i.e. a trait) because it can be
  * changed without loss of information. The converter to do this is the
  * [[PelagoDeviceCross]] operator.
  */
class RelDeviceTypeTraitDef protected () extends RelTraitDef[RelDeviceType] {
  override def getTraitClass: Class[RelDeviceType] = classOf[RelDeviceType]

  override def getSimpleName = "device"

  override def convert(
      planner: RelOptPlanner,
      rel: RelNode,
      toDevice: RelDeviceType,
      allowInfiniteCostConverters: Boolean
  ): RelNode = {
    if ((toDevice eq RelDeviceType.ANY) || rel.getTraitSet.contains(toDevice))
      return rel
    val crossDev = PelagoDeviceCross.create(rel, toDevice)
    var newRel = planner.register(crossDev, rel)
    val newTraitSet = rel.getTraitSet.replace(toDevice)
    if (!(newRel.getTraitSet == newTraitSet))
      newRel = planner.changeTraits(newRel, newTraitSet)
    newRel
//    return PelagoDeviceCross.create(planner.changeTraits(rel, PelagoRel.CONVENTION()), toDevice);
  }

  override def canConvert(
      planner: RelOptPlanner,
      fromTrait: RelDeviceType,
      toDevice: RelDeviceType
  ): Boolean = fromTrait ne toDevice

  override def getDefault: RelDeviceType = RelDeviceType.X86_64
}
