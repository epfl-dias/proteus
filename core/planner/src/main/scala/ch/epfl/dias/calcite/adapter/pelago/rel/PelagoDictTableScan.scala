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

package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits._
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.json4s.JsonDSL._
import org.json4s._

import java.util

/**
  * Relational expression representing a scan of a Pelago dict file.
  */
class PelagoDictTableScan protected (
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
    sql_regex: String,
    attrIndex: Int
) extends LogicalPelagoDictTableScan(
      cluster,
      traitSet,
      table,
      sql_regex,
      attrIndex
    )
    with PelagoRel {
  override def copy(
      traitSet: RelTraitSet,
      inputs: util.List[RelNode]
  ): RelNode = {
    assert(inputs.isEmpty)
    PelagoDictTableScan.create(getCluster, table, regex, attrIndex)
  }

  override def computeBaseSelfCost(
      planner: RelOptPlanner,
      mq: RelMetadataQuery
  ): RelOptCost = {
    super
      .computeSelfCost(planner, mq)
      .multiplyBy(
        10 * (10000 + 2d) / (table.getRowType.getFieldCount.toDouble + 2d)
      )
  }

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    implement(target, null)
  }

  def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    val dictPath = table.unwrap(classOf[PelagoTable]).getPelagoRelName
    val fieldName = table.getRowType.getFieldList.get(attrIndex).getName
    val op = ("operator", "dict-scan") ~
      ("relName", dictPath) ~
      ("attrName", fieldName) ~
      ("regex", regex)

    val binding: Binding = Binding(
      PelagoTable.create(dictPath + ".dict." + fieldName, getRowType),
      getFields(getRowType)
    )
    val ret: (Binding, JValue) = (binding, op)
    ret
  }
}

object PelagoDictTableScan {
  def create(
      cluster: RelOptCluster,
      table: RelOptTable,
      regex: String,
      attrIndex: Int
  ): PelagoDictTableScan = {
    val traitSet = cluster.traitSet
      .replace(PelagoRel.CONVENTION)
      .replace(RelHomDistribution.SINGLE)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(
        RelHetDistributionTraitDef.INSTANCE,
        () => RelHetDistribution.SINGLETON
      )
      .replaceIf(
        RelComputeDeviceTraitDef.INSTANCE,
        () => RelComputeDevice.from(RelDeviceType.X86_64)
      )
      .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
    new PelagoDictTableScan(cluster, traitSet, table, regex, attrIndex)
  }
}
