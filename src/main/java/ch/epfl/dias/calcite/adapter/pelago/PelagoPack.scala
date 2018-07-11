package ch.epfl.dias.calcite.adapter.pelago

import java.util.List

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceTypeTraitDef
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.emitSchema
import org.apache.calcite.plan._
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, RelWriter, SingleRel}
import org.json4s.JValue
import org.json4s.JsonDSL._

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

  override def copy(traitSet: RelTraitSet, inputs: List[RelNode]): PelagoPack = copy(traitSet, inputs.get(0), toPacking)

  def copy(traitSet: RelTraitSet, input: RelNode, packing: RelPacking) = PelagoPack.create(input, packing)

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
    val op = ("operator" , "tuples-to-block")
    val child = getInput.asInstanceOf[PelagoRel].implement
    val childBinding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)

    val json = op ~ ("tupleType", rowType) ~ ("target", getPacking().toString) ~ ("input", childOp) ~ ("trait", getTraitSet.toString)
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

