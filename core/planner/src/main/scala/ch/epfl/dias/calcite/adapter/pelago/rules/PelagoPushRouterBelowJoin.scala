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
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHomDistribution
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.rel.core.Join

object PelagoPushRouterBelowJoin {
  val INSTANCE = new PelagoPushRouterBelowJoin
}

class PelagoPushRouterBelowJoin protected ()
    extends PelagoPushRouterDown(
      classOf[PelagoJoin],
      RelHomDistribution.RANDOM
    ) {

  override def onMatch(call: RelOptRuleCall): Unit = {
    val join: Join = call.rel(0)
    val build = join.getLeft
    val probe = join.getRight
    val new_build = PelagoPushRouterDown.route(build, RelHomDistribution.BRDCST)
    val new_probe = PelagoPushRouterDown.route(probe, trgt)
    call.transformTo(join.copy(null, ImmutableList.of(new_build, new_probe)))
  }
}
