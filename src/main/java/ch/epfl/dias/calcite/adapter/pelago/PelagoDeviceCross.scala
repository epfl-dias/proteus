package ch.epfl.dias.calcite.adapter.pelago

import java.util
import java.util.List

import ch.epfl.dias.calcite.adapter.pelago.metadata.{PelagoRelMdDeviceType, PelagoRelMdDistribution}
import org.apache.calcite.rel.metadata.RelMdDistribution
import org.apache.calcite.rel.{RelDistribution, RelDistributionTraitDef}

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceTypeTraitDef
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema}
import com.google.common.base.Supplier
import org.apache.calcite.plan._
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.SingleRel
import org.json4s.{JValue, JsonAST}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import ch.epfl.dias.emitter.PlanToJSON.{emitAggExpression, emitArg, emitExpression, emitSchema, emit_, getFields}
import ch.epfl.dias.emitter.Binding
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.core.Exchange
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.util.Util

class PelagoDeviceCross protected(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode, val deviceType: RelDeviceType)
      extends SingleRel(cluster, traits, input) with PelagoRel with Converter {
  protected var inTraits: RelTraitSet = input.getTraitSet

  override def explainTerms(pw: RelWriter): RelWriter = {
    val rowCount = input.getCluster.getMetadataQuery.getRowCount(this)
    val bytesPerRow = getRowType.getFieldCount * 4
    val cost = input.getCluster.getPlanner.getCostFactory.makeCost(rowCount * bytesPerRow, rowCount * bytesPerRow, 0).multiplyBy(0.1)

    super.explainTerms(pw)
      .item("trait", getTraitSet.toString).item("intrait", inTraits.toString)
      .item("inputRows", input.getCluster.getMetadataQuery.getRowCount(input))
      .item("cost", cost)
  }

  def getDeviceType() = deviceType;

  override def copy(traitSet: RelTraitSet, inputs: List[RelNode]): PelagoDeviceCross = copy(traitSet, inputs.get(0), deviceType)

  def copy(traitSet: RelTraitSet, input: RelNode, deviceType: RelDeviceType) = PelagoDeviceCross.create(input, deviceType)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = { // Higher cost if rows are wider discourages pushing a project through an
    // exchange.
    val rowCount = mq.getRowCount(this)
    val bytesPerRow = getRowType.getFieldCount * 4
    planner.getCostFactory.makeCost(rowCount * bytesPerRow, rowCount * bytesPerRow, 0).multiplyBy(0.1)

//    if (input.getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == toDevice) planner.getCostFactory.makeHugeCost()
//    else planner.getCostFactory.makeTinyCost
  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = input.estimateRowCount(mq)

  override def implement: (Binding, JValue) = {
    val op = ("operator" , "devcross")
    val child = getInput.asInstanceOf[PelagoRel].implement
    val childBinding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)

    val json = op ~ ("tupleType", rowType) ~ ("target", getDeviceType().toString) ~ ("input", childOp) ~ ("trait", getTraitSet.toString)
    val ret = (childBinding, json)
    ret
  }

  override def getInputTraits: RelTraitSet = inTraits

  override def getTraitDef: RelTraitDef[_ <: RelTrait] = RelDeviceTypeTraitDef.INSTANCE
}

object PelagoDeviceCross {
  def create(input: RelNode, toDevice: RelDeviceType): PelagoDeviceCross = {
    val cluster = input.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(toDevice)
    new PelagoDeviceCross(input.getCluster, traitSet, input, toDevice)
  }
}
