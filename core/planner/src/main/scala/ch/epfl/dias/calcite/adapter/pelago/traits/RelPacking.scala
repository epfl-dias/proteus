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

/**
  * Description of the target packing of a relational expression.
  */
object RelPacking {
  val Packed = new RelPacking("packed")
  val UnPckd = new RelPacking("unpckd")
}

class RelPacking protected (val p: String) extends PelagoTrait {
  override def toString: String = p
  override def getTraitDef: RelPackingTraitDef = RelPackingTraitDef.INSTANCE
  override def satisfies(`trait`: RelTrait): Boolean = `trait` eq this
  override def register(planner: RelOptPlanner): Unit = {}
}
