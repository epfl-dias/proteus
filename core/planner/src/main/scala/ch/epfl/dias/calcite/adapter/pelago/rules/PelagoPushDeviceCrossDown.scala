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

import ch.epfl.dias.calcite.adapter.pelago.traits.RelDeviceType
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterRule
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoAggregate
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoFilter
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoJoin
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoProject
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoSort
import org.apache.calcite.plan.RelOptRule

import scala.collection.JavaConverters._

object PelagoPushDeviceCrossDown {
  val RULES: Array[PelagoPushDeviceCrossDown] = Array(
    new PelagoPushDeviceCrossDown(classOf[PelagoAggregate]),
    new PelagoPushDeviceCrossDown(classOf[PelagoFilter]),
    new PelagoPushDeviceCrossDown(classOf[PelagoProject]),
    new PelagoPushDeviceCrossDown(classOf[PelagoSort]),
    new PelagoPushDeviceCrossDown(classOf[PelagoJoin])
  )
}
class PelagoPushDeviceCrossDown protected (val op: Class[_ <: RelNode])
    extends ConverterRule(
      op,
      RelDeviceType.X86_64,
      RelDeviceType.NVPTX,
      "PPDCD" + op.getName
    ) {

  protected def cross(rel: RelNode): RelNode =
    RelOptRule.convert(rel, RelDeviceType.NVPTX)

  override def convert(rel: RelNode): RelNode = {
    val inps = rel.getInputs.asScala
      .map(this.cross)
      .toList
      .asJava
    rel.copy(null, inps)
  }
}
