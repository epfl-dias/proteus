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

object RelDeviceType {
  val X86_64 = new RelDeviceType("X86_64")
  val NVPTX = new RelDeviceType("NVPTX")
  val ANY = new RelDeviceType("anydev")
}

/**
  * Description of the target device of a relational expression.
  */
class RelDeviceType protected (val dev: String) extends PelagoTrait {
  override def toString: String = dev

  override def getTraitDef: RelDeviceTypeTraitDef =
    RelDeviceTypeTraitDef.INSTANCE

  override def satisfies(`trait`: RelTrait): Boolean = {
    (this eq RelDeviceType.ANY) || (`trait` eq this)
//    return (trait == this) || (trait == ANY); //(this == ANY) ||
  }

  def getMemBW: Double =
    (if (this eq RelDeviceType.X86_64) 100.0 / 10
     else 900) * 1024.0 * 1024 * 1024

  override def register(planner: RelOptPlanner): Unit = {}
}
