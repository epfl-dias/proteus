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

import ch.epfl.dias.calcite.adapter.pelago.rel.{PelagoFilter, PelagoProject}
import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelDeviceType,
  RelDeviceTypeTraitDef,
  RelHomDistribution
}
import org.apache.calcite.plan.RelOptRule.{any, convert, operand}
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall}
import org.apache.calcite.rel.RelNode

import scala.collection.JavaConverters._

object PelagoPushRouterDown {
  val RULES: Array[RelOptRule] = Array[RelOptRule](
    new PelagoPushRouterDown(classOf[PelagoFilter], RelHomDistribution.RANDOM),
    new PelagoPushRouterDown(classOf[PelagoProject], RelHomDistribution.RANDOM),
    new PelagoPushRouterDown(classOf[PelagoFilter], RelHomDistribution.BRDCST),
    new PelagoPushRouterDown(classOf[PelagoProject], RelHomDistribution.BRDCST),
    PelagoPushRouterBelowJoin.INSTANCE,
    PelagoPushRouterBelowAggregate.INSTANCE
  )

  def route(
      input: RelNode,
      trgt: RelHomDistribution
  ): RelNode =
    convert(
      input,
      input.getTraitSet
        .replace(trgt)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
    )
}

class PelagoPushRouterDown protected (
    val op: Class[_ <: RelNode],
    val trgt: RelHomDistribution
) extends RelOptRule(operand(op, any), "PPRD" + op.getName + trgt.toString) {

  protected def route(input: RelNode): RelNode =
    PelagoPushRouterDown.route(input, trgt)

  override def matches(call: RelOptRuleCall): Boolean = {
    call
      .rel(0)
      .asInstanceOf[RelNode]
      .getTraitSet
      .containsIfApplicable(RelHomDistribution.SINGLE)
  }

  override def onMatch(call: RelOptRuleCall): Unit = {
    val rel = call.rel(0).asInstanceOf[RelNode]
    call.transformTo(
      rel.copy(
        null,
        rel.getInputs.asScala.map(this.route).asJava
      )
    )
  }
}
