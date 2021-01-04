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

import ch.epfl.dias.calcite.adapter.pelago.rel.{
  PelagoDeviceCross,
  PelagoRouter,
  PelagoSplit,
  PelagoUnion
}
import ch.epfl.dias.calcite.adapter.pelago.traits.RelPacking
import org.apache.calcite.plan.RelOptRule.{any, convert, operand}
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall}
import org.apache.calcite.rel.RelNode

import scala.collection.JavaConverters._

object PelagoPackTransfers {
  val RULES: Array[RelOptRule] = Array[RelOptRule](
    new PelagoPackTransfers(classOf[PelagoUnion]),
    new PelagoPackTransfers(classOf[PelagoRouter]),
    new PelagoPackTransfers(classOf[PelagoSplit]),
    new PelagoPackTransfers(classOf[PelagoDeviceCross])
  )
}
class PelagoPackTransfers protected (val op: Class[_ <: RelNode])
    extends RelOptRule(operand(op, any), "PPT" + op.getName) {

  protected def pack(rel: RelNode): RelNode = convert(rel, RelPacking.Packed)

  override def onMatch(call: RelOptRuleCall): Unit = {
    val rel: RelNode = call.rel(0)
    val inps =
      rel.getInputs.asScala
        .map(this.pack)
        .toList
        .asJava
    call.transformTo(rel.copy(null, inps))
  }
}
