package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoRules
import ch.epfl.dias.emitter.{Binding, PlanToJSON}
import org.apache.calcite.linq4j.tree.Primitive
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.TableScan
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.`type`.RelDataTypeFactory
import org.apache.calcite.rel.`type`.RelDataTypeField
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import org.json4s.JsonAST

import scala.Tuple2
import java.util

import com.google.common.base.Supplier
import org.apache.calcite.rex.RexInputRef
import org.json4s

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
import ch.epfl.dias.emitter.PlanToJSON._

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

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("fields", Primitive.asList(fields)).item("traits", getTraitSet.toString)

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

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = { // Multiply the cost by a factor that makes a scan more attractive if it
    // has significantly fewer fields than the original scan.
    //
    // The "+ 2D" on top and bottom keeps the function fairly smooth.
    //
    // For example, if table has 3 fields, project has 1 field,
    // then factor = (1 + 2) / (3 + 2) = 0.6
    super.computeSelfCost(planner, mq).multiplyBy((fields.length.toDouble * 10000 + 2D) / (table.getRowType.getFieldCount.toDouble + 2D))
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

  def implement: (Binding, JValue) = {
    val op = ("operator" , "scan")
    //TODO Cross-check: 0: schemaName, 1: tableName (?)
    val srcName  = getPelagoRelName //s.getTable.getQualifiedName.get(1)

    val rowType  = emitSchema(srcName, getRowType)
    val linehint = getLineHint.longValue

    val pluginfo = getPluginInfo.asScala.+(("name", srcName)).+(("projections", rowType))
//      .+(("type", "bincol")).+(("sizeInFile", false)) //FIXME: to be removed!!! (removing it requires unpack operator)
    val plugin   = Extraction.decompose(pluginfo)

    val json : JValue = op ~
      ("gpu"      , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX))       ~
      ("plugin"   , plugin  )
    val binding: Binding = Binding(srcName, getFields(getRowType))
    val ret: (Binding, JValue) = (binding,json)
    ret
  }

  def getDistribution: RelDistribution = {
    pelagoTable.getDistribution
  }

  def getDeviceType: RelDeviceType = {
    pelagoTable.getDeviceType
  }
}

object PelagoTableScan {
  def create(cluster: RelOptCluster, table: RelOptTable, pelagoTable: PelagoTable, fields: Array[Int]) = {
      val traitSet = cluster.traitSet.replace(PelagoRel.CONVENTION)
          .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier[RelDistribution]() {
              override def get: RelDistribution = {
                return pelagoTable.getDistribution
              }
          }).replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier[RelDeviceType  ]() {
              override def get: RelDeviceType = {
                return pelagoTable.getDeviceType
              }
          }).replaceIf(RelPackingTraitDef   .INSTANCE, new Supplier[RelPacking     ]() {
            override def get: RelPacking = {
              return pelagoTable.getPacking
            }
          });
    new PelagoTableScan(cluster, traitSet, table, pelagoTable, fields)
  }
}