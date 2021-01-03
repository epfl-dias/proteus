package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelComputeDevice, RelComputeDeviceTraitDef, RelDeviceType, RelDeviceTypeTraitDef, RelHetDistribution, RelHetDistributionTraitDef, RelHomDistribution, RelPacking, RelPackingTraitDef}
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.json4s.JsonDSL._
import org.json4s._

import java.util

/**
  * Relational expression representing a scan of a Pelago file.
  *
  * Based on:
  * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTableScan.java
  *
  * <p>Like any table scan, it serves as a leaf node of a query tree.</p>
  */
class PelagoDictTableScan protected (cluster: RelOptCluster, traitSet: RelTraitSet, table: RelOptTable,
                                     sql_regex: String, val attrIndex: Int)
      extends TableScan(cluster, traitSet, table) with PelagoRel {
  val regex = sql_regex
    .replaceAll("\\\\", "\\\\")
    .replaceAll("\\*", "\\*")
    .replaceAll("\\.", "\\.")
    .replaceAll("\\^", "\\^")
    .replaceAll("\\$", "\\$")
    .replaceAll("%", ".*")
    .replaceAll("_", ".+")

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): RelNode = {
    assert(inputs.isEmpty)
    PelagoDictTableScan.create(getCluster, table, regex, attrIndex)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("regex", regex).item("traits", getTraitSet.toString)

  override def deriveRowType: RelDataType = {
    getCluster.getTypeFactory.builder
      .add(table.getRowType.getFieldList.get(attrIndex).getName, table.getRowType.getFieldList.get(attrIndex).getType)
      .build()
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    super.computeSelfCost(planner, mq).multiplyBy(10*(10000 + 2D) / (table.getRowType.getFieldCount.toDouble + 2D))
  }

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    return implement(target, null)
  }

  def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
//    assert(alias == null);//this operator does not provide arbitrary projections
    val dictPath = table.unwrap(classOf[PelagoTable]).getPelagoRelName
    val fieldName = table.getRowType.getFieldList.get(attrIndex).getName
    val op = ("operator", "dict-scan" ) ~
      ("relName"        , dictPath    ) ~
      ("attrName"       , fieldName   ) ~
      ("regex"          , regex       )

    val binding: Binding = Binding(PelagoTable.create(dictPath + "$dict$" + fieldName, getRowType), getFields(getRowType))
    val ret: (Binding, JValue) = (binding, op)
    ret
  }

  def getHomDistribution: RelHomDistribution = {
    RelHomDistribution.SINGLE
  }

  def getDeviceType: RelDeviceType = {
    RelDeviceType.X86_64
  }
}

object PelagoDictTableScan {
  def create(cluster: RelOptCluster, table: RelOptTable, regex: String, attrIndex: Int) = {
    val mq = cluster.getMetadataQuery
    val traitSet = cluster.traitSet.replace(PelagoRel.CONVENTION)
      .replace(RelHomDistribution.SINGLE)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelHetDistributionTraitDef.INSTANCE, () => RelHetDistribution.SINGLETON)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(RelDeviceType.X86_64))
      .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd);
    new PelagoDictTableScan(cluster, traitSet, table, regex, attrIndex)
  }
}