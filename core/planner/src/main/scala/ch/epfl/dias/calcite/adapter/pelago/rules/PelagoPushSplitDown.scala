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
import ch.epfl.dias.calcite.adapter.pelago.traits._
import org.apache.calcite.plan.RelOptRule.{any, convert, operand}
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall}
import org.apache.calcite.rel.RelNode

import scala.collection.JavaConverters._

object PelagoPushSplitDown {
  val RULES: Array[RelOptRule] = Array[RelOptRule](
    new PelagoPushSplitDown(classOf[PelagoFilter], RelHetDistribution.SPLIT),
    new PelagoPushSplitDown(
      classOf[PelagoFilter],
      RelHetDistribution.SPLIT_BRDCST
    ),
    new PelagoPushSplitDown(classOf[PelagoProject], RelHetDistribution.SPLIT),
    new PelagoPushSplitDown(
      classOf[PelagoProject],
      RelHetDistribution.SPLIT_BRDCST
    ),
    PelagoPushSplitBelowJoin.INSTANCE,
    PelagoPushSplitBelowAggregate.INSTANCE
  )

  def split(rel: RelNode, distr: RelHetDistribution): RelNode =
    convert(
      rel,
      rel.getTraitSet
        .replace(distr)
        .replace(
          if (
            rel.getTraitSet
              .getTrait(RelDeviceTypeTraitDef.INSTANCE) eq RelDeviceType.NVPTX
          ) RelComputeDevice.NVPTX
          else RelComputeDevice.X86_64
        )
        .replace(RelSplitPoint.of(rel))
    )
}
class PelagoPushSplitDown protected (
    val op: Class[_ <: RelNode],
    val distr: RelHetDistribution
) extends RelOptRule(operand(op, any), "PPSD" + op.getName + distr.toString) {

  protected def split(rel: RelNode): RelNode =
    PelagoPushSplitDown.split(rel, distr)

  override def onMatch(call: RelOptRuleCall): Unit = {
    val rel: RelNode = call.rel(0)
    call.transformTo(
      rel.copy(
        null,
        rel.getInputs.asScala.map(this.split).asJava
      )
    )
  }
}
