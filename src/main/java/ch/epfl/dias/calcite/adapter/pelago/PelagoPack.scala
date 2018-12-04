package ch.epfl.dias.calcite.adapter.pelago

import java.util

import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.plan._
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, RelWriter, SingleRel}
import org.apache.calcite.rex.RexInputRef
import org.json4s.JValue
import org.json4s.JsonAST.JObject
import org.json4s.JsonDSL._

import scala.collection.JavaConverters._

class PelagoPack protected(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode, val toPacking: RelPacking)
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

  def getPacking() = toPacking;

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): PelagoPack = copy(traitSet, inputs.get(0), toPacking)

  def copy(traitSet: RelTraitSet, input: RelNode, packing: RelPacking) = PelagoPack.create(input, packing)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = { // Higher cost if rows are wider discourages pushing a project through an
    // exchange.
    val rf = {
      if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) 0.001
      else 1
    }
    val rowCount = mq.getRowCount(this)
    val bytesPerRow = getRowType.getFieldCount * 4
    planner.getCostFactory.makeCost(rowCount * bytesPerRow, rowCount * bytesPerRow, 0).multiplyBy(rf * 100)

//    if (input.getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == toDevice) planner.getCostFactory.makeHugeCost()
//    else planner.getCostFactory.makeTinyCost
//    planner.getCostFactory.makeZeroCost
  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = Math.ceil(input.estimateRowCount(mq) / (1024))

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    val op = ("operator" , "tuples-to-block")
    val child = getInput.asInstanceOf[PelagoRel].implement(target)
    val childBinding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)

    val projs = getRowType.getFieldList.asScala.zipWithIndex.map{
      f => {
        emitExpression(RexInputRef.of(f._2, getInput.getRowType), List(childBinding)).asInstanceOf[JObject]
      }
    }

    val json = op ~
      ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("projections", projs) ~ ("input", childOp) ~ ("trait", getTraitSet.toString)
    val ret = (childBinding, json)
    ret
  }

  override def getInputTraits: RelTraitSet = inTraits

  override def getTraitDef: RelTraitDef[_ <: RelTrait] = RelPackingTraitDef.INSTANCE
}

object PelagoPack {
  def create(input: RelNode, toPacking: RelPacking): PelagoPack = {
    val cluster = input.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(toPacking)
    new PelagoPack(input.getCluster, traitSet, input, toPacking)
  }
}

