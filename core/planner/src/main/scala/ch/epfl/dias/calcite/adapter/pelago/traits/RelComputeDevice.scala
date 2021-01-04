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

import org.apache.calcite.plan.{RelOptPlanner, RelTrait}
import org.apache.calcite.rel.RelNode

/**
  * TODO: should we convert it into a RelMultipleTrait ? Does a RelMultipleTrait has *ANY* of the values or all?
  */
object RelComputeDevice {
  val NONE = new RelComputeDevice("none")
  val X86_64 = new RelComputeDevice("cX86_64")
  val NVPTX = new RelComputeDevice("cNVPTX")
  val X86_64NVPTX = new RelComputeDevice("cNV+X86")
  def from(dev: RelDeviceType): RelComputeDevice = {
    if (dev eq RelDeviceType.X86_64) return RelComputeDevice.X86_64
    else if (dev eq RelDeviceType.NVPTX) return RelComputeDevice.NVPTX
    assert(false)
    RelComputeDevice.X86_64NVPTX
  }
  def from(input: RelNode): RelComputeDevice = {
    val `trait` = input.getTraitSet
    val dev = `trait`.getTrait(RelDeviceTypeTraitDef.INSTANCE)
    val comp = RelComputeDevice.from(input, isCompute = false)
    RelComputeDevice.from(
      List(RelComputeDevice.from(dev), comp)
    )
  }
  def from(input: RelNode, isCompute: Boolean): RelComputeDevice = {
    if (isCompute) return from(input)
    val `trait` = input.getTraitSet
    `trait`.getTrait(RelComputeDeviceTraitDef.INSTANCE)
  }
  def from(
      relComputeDeviceStream: List[RelComputeDevice]
  ): RelComputeDevice = {
    val devs = relComputeDeviceStream.distinct
      .filter((e: RelComputeDevice) => e ne NONE)
    if (devs.isEmpty) return NONE
    if (devs.size == 1) return devs.head
    X86_64NVPTX //NOTE: do not forget to update this if you add devices!
  }
}

class RelComputeDevice protected (val computeTypes: String)
    extends PelagoTrait {

  override def toString: String = computeTypes

  override def getTraitDef: RelComputeDeviceTraitDef =
    RelComputeDeviceTraitDef.INSTANCE

  override def satisfies(`trait`: RelTrait): Boolean = {
    if (`trait` eq this)
      return true // everything satisfies itself... (Singleton)
    if (this eq RelComputeDevice.NONE)
      return true // no processing can be considered as any processing
    if (`trait` eq RelComputeDevice.NONE)
      return false // only itself satisfies NONE
    if (`trait` eq RelComputeDevice.X86_64NVPTX) { // for now, returned expression is always true, but leave it here for future-proofness.
      return (this eq RelComputeDevice.X86_64) || (this eq RelComputeDevice.NVPTX)
    }
    false
  }

  override def register(planner: RelOptPlanner): Unit = {}
}
