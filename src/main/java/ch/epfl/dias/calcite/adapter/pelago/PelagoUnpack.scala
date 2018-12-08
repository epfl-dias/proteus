package ch.epfl.dias.calcite.adapter.pelago

import java.util

import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, getFields}
import org.apache.calcite.rel.{RelDistribution, RelDistributions}
import org.apache.calcite.rex.RexInputRef

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceTypeTraitDef
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.emitSchema
import org.apache.calcite.plan._
import org.apache.calcite.rel.{RelNode, RelWriter, SingleRel}
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.json4s.JValue
import org.json4s.JsonDSL._
import org.json4s.{JValue, JsonAST}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._

class PelagoUnpack protected(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode, val toPacking: RelPacking)
      extends SingleRel(cluster, traits, input) with PelagoRel with Converter {
  protected var inTraits: RelTraitSet = input.getTraitSet

  override def explainTerms(pw: RelWriter): RelWriter = {
    super.explainTerms(pw)
      .item("trait", getTraitSet.toString)
  }

  def getPacking() = toPacking;

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): PelagoUnpack = copy(traitSet, inputs.get(0), toPacking)

  def copy(traitSet: RelTraitSet, input: RelNode, packing: RelPacking) = PelagoUnpack.create(input, packing)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = { // Higher cost if rows are wider discourages pushing a project through an
    val rf = {
      if (!getTraitSet.containsIfApplicable(RelDistributions.SINGLETON)) {
        if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) 0.0001
        else 0.001
      } else if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
        0.001
      } else {
        0.1
      }
    }
    // exchange.
    val rowCount = mq.getRowCount(this)
    val bytesPerRow = getRowType.getFieldCount * 4
    planner.getCostFactory.makeCost(rowCount, rowCount * bytesPerRow * rf * 100, 0)

//    if (input.getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == toDevice) planner.getCostFactory.makeHugeCost()
//    else planner.getCostFactory.makeTinyCost
  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = input.estimateRowCount(mq) * (1024)

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    val op = ("operator", "block-to-tuples")
    val child = getInput.asInstanceOf[PelagoRel].implement(getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE))
    val childBinding: Binding = child._1
    val childOp = child._2
    val alias   = childBinding.rel  //"unpack" + getId

    val projs = getRowType.getFieldList.asScala.zipWithIndex.map {
      f => {
        emitExpression(RexInputRef.of(f._2, getInput.getRowType), List(childBinding))
      }
    }
    val json = op ~
      ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("projections", projs                                                 ) ~
      ("input"      , childOp                                               )

    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding, json)
    ret
  }

  override def getInputTraits: RelTraitSet = inTraits

  override def getTraitDef: RelTraitDef[_ <: RelTrait] = RelPackingTraitDef.INSTANCE
}

object PelagoUnpack {
  def create(input: RelNode, toPacking: RelPacking): PelagoUnpack = {
    val cluster = input.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(toPacking)
    new PelagoUnpack(input.getCluster, traitSet, input, toPacking)
  }
}

