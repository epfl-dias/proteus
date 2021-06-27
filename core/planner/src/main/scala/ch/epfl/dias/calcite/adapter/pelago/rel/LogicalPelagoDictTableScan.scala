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

import ch.epfl.dias.calcite.adapter.pelago.traits._
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.metadata.RelMetadataQuery

import java.util

/**
  * Relational expression representing a scan of a Pelago dict file.
  */
class LogicalPelagoDictTableScan protected (
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    table: RelOptTable,
    sql_regex: String,
    val attrIndex: Int
) extends TableScan(cluster, traitSet, util.List.of(), table) {
  lazy val regex: String = sql_regex
    .replaceAll("\\\\", "\\\\")
    .replaceAll("\\*", "\\*")
    .replaceAll("\\.", "\\.")
    .replaceAll("\\^", "\\^")
    .replaceAll("\\$", "\\$")
    .replaceAll("%", ".*")
    .replaceAll("_", ".+")

  override def copy(
      traitSet: RelTraitSet,
      inputs: util.List[RelNode]
  ): RelNode = {
    assert(inputs.isEmpty)
    LogicalPelagoDictTableScan.create(getCluster, table, sql_regex, attrIndex)
  }

  override def explainTerms(pw: RelWriter): RelWriter =
    super
      .explainTerms(pw)
      .item("regex", regex)
      .item("traits", getTraitSet.toString)

  override def deriveRowType: RelDataType = {
    getCluster.getTypeFactory.builder
      .add(
        table.getRowType.getFieldList.get(attrIndex).getName,
        table.getRowType.getFieldList.get(attrIndex).getType
      )
      .build()
  }

  override def computeSelfCost(
      planner: RelOptPlanner,
      mq: RelMetadataQuery
  ): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  def getHomDistribution: RelHomDistribution = {
    RelHomDistribution.SINGLE
  }

  def getDeviceType: RelDeviceType = {
    RelDeviceType.X86_64
  }
}

object LogicalPelagoDictTableScan {
  def create(
      cluster: RelOptCluster,
      table: RelOptTable,
      regex: String,
      attrIndex: Int
  ): LogicalPelagoDictTableScan = {
    val traitSet = cluster.traitSet
      .replace(Convention.NONE)
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
    new LogicalPelagoDictTableScan(cluster, traitSet, table, regex, attrIndex)
  }
}
