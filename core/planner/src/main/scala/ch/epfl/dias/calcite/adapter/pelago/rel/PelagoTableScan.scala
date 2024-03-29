package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.costs.CostModel
import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelComputeDevice, RelComputeDeviceTraitDef, RelDeviceType, RelDeviceTypeTraitDef, RelHomDistribution, RelPacking, RelPackingTraitDef}
import ch.epfl.dias.emitter.PlanToJSON._
import ch.epfl.dias.emitter.{Binding, PlanToJSON}
import org.apache.calcite.linq4j.tree.Primitive
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.json4s.JsonDSL._
import org.json4s._

import java.util
import scala.collection.JavaConverters._

/**
  * Relational expression representing a scan of a Pelago file.
  *
  * Based on:
  * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTableScan.java
  *
  * <p>Like any table scan, it serves as a leaf node of a query tree.</p>
  */
class PelagoTableScan protected (cluster: RelOptCluster, traitSet: RelTraitSet, table: RelOptTable, val pelagoTable: PelagoTable, val fields: Array[Int])
      extends TableScan(cluster, traitSet, table) with PelagoRel {
  assert(pelagoTable != null)

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): RelNode = {
    assert(inputs.isEmpty)
    PelagoTableScan.create(getCluster, table, pelagoTable, fields)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("fields", Primitive.asList(fields))

  override def deriveRowType: RelDataType = {
    val fieldList = table.getRowType.getFieldList
    val builder = getCluster.getTypeFactory.builder
    for (field <- fields) {
      builder.add(fieldList.get(field))
    }
    builder.build
  }

  override def register(planner: RelOptPlanner): Unit = {
//    for (rule <- PelagoRules.RULES) {
//      planner.addRule(rule)
//    }
    //    planner.addRule(PelagoProjectTableScanRule.INSTANCE);
  }


  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    // Multiply the cost by a factor that makes a scan more attractive if it
    // has significantly fewer fields than the original scan.
    val s = super.computeSelfCost(planner, mq)
    planner.getCostFactory.makeCost(
      s.getRows,
      s.getCpu * (fields.length.toDouble * 10000 + 2D) / (table.getRowType.getFieldCount.toDouble + 2D) + fields.length * 1e8,
      s.getRows * fields.length.toDouble * 2 * 1e8
    )
  }

  def getPelagoRelName: String = pelagoTable.getPelagoRelName

  def getPluginInfo: util.Map[String, _] = pelagoTable.getPluginInfo

  def getLineHint: Long = pelagoTable.getLineHint

//  override def implement: (Binding, JValue) = {
//    val op = ("operator" , "scan")
//    //TODO Cross-check: 0: schemaName, 1: tableName (?)
//    val srcName  = getPelagoRelName //s.getTable.getQualifiedName.get(1)
//    val rowType  = emitSchema(srcName, getRowType)
//    val plugin   = Extraction.decompose(getPluginInfo.asScala)
//    val linehint = getLineHint.longValue
//
//    val json : JValue = op ~ ("tupleType", rowType) ~ ("name", srcName) ~ ("plugin", plugin) ~ ("linehint", linehint)
//    val binding: Binding = Binding(srcName, getFields(getRowType))
//    val ret: (Binding, JValue) = (binding,json)
//    ret
//  }

//  override def implement: (Binding, JValue) = {
//    val op = ("operator", "block-to-tuples")
//    val child = implementScan
//    val childBinding: Binding = child._1
//    val childOp = child._2
//    val alias = childBinding.rel
//
//
//    val projs = getRowType.getFieldList.asScala.zipWithIndex.map {
//      f => {
//        emitExpression(RexInputRef.of(f._2, /*child.*/ getRowType), List(childBinding))
//      }
//    }
//
//    val json = op ~
//      ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
//      ("projections", projs                                                 ) ~
//      ("input"      , childOp                                               )
//    val binding: Binding = Binding(alias, getFields(getRowType))
//    val ret: (Binding, JValue) = (binding, json)
//    ret
//  }

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    return implement(target, null)
  }

  def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
//    assert(alias == null);//this operator does not provide arbitrary projections
    val op = ("operator" , "scan")
    //TODO Cross-check: 0: schemaName, 1: tableName (?)
    val srcName  = getPelagoRelName //s.getTable.getQualifiedName.get(1)

    PlanToJSON.dictEncoded = getPluginInfo.get("type").toString.contains("block")
    val rowType  = emitSchema(pelagoTable, getRowType)
    val linehint = getLineHint.longValue

    val tableBinding: Binding = Binding(pelagoTable, table.getRowType.getFieldList.asScala.toList)

//    val projs = fields.map{
//      f => {
//        emitExpression(RexInputRef.of(f, table.getRowType), List(tableBinding)).asInstanceOf[JObject]
//      }
//    }.toList
//
//    val schema = table.getRowType.getFieldList.asScala.map{
//      f => {
//        emitExpression(RexInputRef.of(f.getIndex, table.getRowType), List(tableBinding)).asInstanceOf[JObject]
//      }
//    }.toList


    val plugin = Extraction.decompose(getPluginInfo.asScala).asInstanceOf[JObject] ~
      ("name"       , srcName) ~
      ("projections", rowType) ~
      ("schema"     , emitSchema(pelagoTable, table.getRowType, true, false, true))

    val json : JValue = op ~
      ("gpu"      , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX))       ~
      ("plugin"   , plugin  )
    PlanToJSON.dictEncoded = false

    val binding: Binding = Binding(pelagoTable, getFields(getRowType))
    val ret: (Binding, JValue) = (binding,json)
    ret
  }

  def getHomDistribution: RelHomDistribution = {
    pelagoTable.getHomDistribution
  }

  def getDeviceType: RelDeviceType = {
    pelagoTable.getDeviceType
  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = table.getRowCount / (if (pelagoTable.getPacking == RelPacking.Packed) CostModel.blockSize else 1)
}

object PelagoTableScan {
  def create(cluster: RelOptCluster, table: RelOptTable, pelagoTable: PelagoTable, fields: Array[Int]) = {
      val traitSet = cluster.traitSet.replace(PelagoRel.CONVENTION)
        .replace(pelagoTable.getHomDistribution)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => pelagoTable.getDeviceType)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => pelagoTable.getPacking)
        .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => if (pelagoTable.getPacking == RelPacking.Packed) RelComputeDevice.NONE else RelComputeDevice.from(pelagoTable.getDeviceType));
    new PelagoTableScan(cluster, traitSet, table, pelagoTable, fields)
  }
}