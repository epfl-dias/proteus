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

import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoJoin
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHetDistribution
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptRuleCall

object PelagoPushSplitBelowJoin { val INSTANCE = new PelagoPushSplitBelowJoin }

class PelagoPushSplitBelowJoin protected ()
    extends PelagoPushSplitDown(classOf[PelagoJoin], RelHetDistribution.SPLIT) {
  override def onMatch(call: RelOptRuleCall): Unit = {
    val join: PelagoJoin = call.rel(0)
    val build = join.getLeft
    val probe = join.getRight
    val new_build =
      PelagoPushSplitDown.split(build, RelHetDistribution.SPLIT_BRDCST)
    val new_probe = split(probe)
    call.transformTo(join.copy(null, ImmutableList.of(new_build, new_probe)))
  }
}
