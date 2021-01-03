package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelDeviceType, RelDeviceTypeTraitDef, RelHomDistributionTraitDef}
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.plan._
import org.apache.calcite.prepare.Prepare
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.TableModify
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.{RexInputRef, RexNode}
import org.json4s.JsonDSL._
import org.json4s.{JObject, JValue}

import java.util
import scala.collection.JavaConverters._

/**
 * Relational expression representing a scan of a table in a Pelago data source.
 */
class PelagoTableModify protected(cluster: RelOptCluster, traitSet: RelTraitSet,
                                  table: RelOptTable, schema: Prepare.CatalogReader, input: RelNode,
                                  operation: TableModify.Operation, updateColumnList: java.util.List[String],
                                  sourceExpressionList: java.util.List[RexNode], flattened: Boolean)
  extends TableModify(cluster, traitSet,
    table, schema, input: RelNode,
    operation, updateColumnList,
    sourceExpressionList, flattened) with PelagoRel {

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): RelNode = copy(traitSet, inputs.get(0))

  def copy(traitSet: RelTraitSet, input: RelNode): RelNode = PelagoTableModify.create(getTable, getCatalogReader, input,
    getOperation, getUpdateColumnList, getSourceExpressionList, isFlattened);

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    super.computeSelfCost(planner, mq).multiplyBy(getRowType.getFieldCount.toDouble * 0.1)
  }

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    val op = ("operator", "insert-into")
    val alias = getTable.unwrap(classOf[PelagoTable])
    val child = getInput.asInstanceOf[PelagoRel].implement(RelDeviceType.X86_64)
    val childBinding = child._1
    val childOp = child._2

    val exprs = table.getRowType
    val exprsJS: JValue = exprs.getFieldList.asScala.zipWithIndex.map {
      e => {
        val reg_as = ("attrName", table.getRowType.getFieldNames.get(e._2)) ~ ("relName", alias.alias)
        emitExpression(RexInputRef.of(e._1.getIndex, table.getRowType), List(childBinding), this).asInstanceOf[JObject] ~ ("register_as", reg_as)
      }
    }

    val json = op ~
      ("name", alias.alias) ~
      ("e"    , exprsJS                                              ) ~
      ("plugin", ("type", alias.plugin.asScala("type").toString)) ~
      ("input", childOp)

    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding, json)
    ret
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    val c = super.computeSelfCost(planner, mq)
    planner.getCostFactory.makeCost(c.getRows, c.getCpu * 1e15, c.getIo)
  }
}

object PelagoTableModify {
  def create(table: RelOptTable, schema: Prepare.CatalogReader, input: RelNode, operation: TableModify.Operation, updateColumnList: util.List[String], sourceExpressionList: util.List[RexNode], flattened: Boolean): PelagoTableModify = {
    val cluster: RelOptCluster = input.getCluster
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION)
      .replaceIf(RelHomDistributionTraitDef.INSTANCE, () => cluster.getMetadataQuery.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => cluster.getMetadataQuery.asInstanceOf[PelagoRelMetadataQuery].deviceType(input))
    new PelagoTableModify(cluster, traitSet, table, schema, input, operation, updateColumnList, sourceExpressionList, flattened)
  }
}

