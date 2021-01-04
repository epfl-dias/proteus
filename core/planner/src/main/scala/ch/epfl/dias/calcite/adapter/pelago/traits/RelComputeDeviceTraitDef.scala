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

import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.{RelOptPlanner, RelTraitDef}
import org.apache.calcite.rel.RelNode

import scala.collection.JavaConverters._

object RelComputeDeviceTraitDef { val INSTANCE = new RelComputeDeviceTraitDef }

class RelComputeDeviceTraitDef protected ()
    extends RelTraitDef[RelComputeDevice] {

  override def getTraitClass: Class[RelComputeDevice] =
    classOf[RelComputeDevice]

  override def getSimpleName = "compute"

  override def convert(
      planner: RelOptPlanner,
      rel: RelNode,
      toDevice: RelComputeDevice,
      allowInfiniteCostConverters: Boolean
  ): RelNode = { //    if (rel.getTraitSet().getTrait(INSTANCE).satisfies(toDevice)) return rel;
    val inputs = rel.getInputs
    if (inputs.isEmpty) return null
    val dev =
      if (toDevice eq RelComputeDevice.NVPTX) RelDeviceType.NVPTX
      else RelDeviceType.X86_64
    val b = ImmutableList.builder[RelNode]
    for (inp <- inputs.asScala) {
      b.add(planner.changeTraits(inp, inp.getTraitSet.replace(dev)))
    }
    var newRel = rel.copy(null, b.build)
    newRel = planner.register(newRel, rel)
    if (!newRel.getTraitSet.contains(toDevice)) return null
    val traitSet = rel.getTraitSet.replace(toDevice)
    if (!(newRel.getTraitSet == traitSet))
      newRel = planner.changeTraits(newRel, traitSet)
    newRel
  }

  override def canConvert(
      planner: RelOptPlanner,
      fromTrait: RelComputeDevice,
      toDevice: RelComputeDevice
  ): Boolean = { //See comment in convert(...)
    false //toDevice != RelComputeDevice.X86_64NVPTX && toDevice != RelComputeDevice.NONE;//fromTrait.satisfies(toDevice);

  }

  override def getDefault: RelComputeDevice = RelComputeDevice.X86_64NVPTX
}
