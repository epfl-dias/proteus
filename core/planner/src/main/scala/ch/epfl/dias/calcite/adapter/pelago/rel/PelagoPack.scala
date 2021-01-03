package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelComputeDevice, RelComputeDeviceTraitDef, RelDeviceType, RelHomDistribution, RelPacking, RelPackingTraitDef}
import ch.epfl.dias.emitter.Binding
import org.apache.calcite.plan._
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, SingleRel}
import org.json4s.JValue
import org.json4s.JsonDSL._

import java.util

class PelagoPack protected(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode, val toPacking: RelPacking)
      extends SingleRel(cluster, traits, input) with PelagoRel with Converter {
  protected var inTraits: RelTraitSet = input.getTraitSet

  def getPacking() = toPacking;

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): PelagoPack = copy(traitSet, inputs.get(0), toPacking)

  def copy(traitSet: RelTraitSet, input: RelNode, packing: RelPacking) = PelagoPack.create(input, packing)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    val rf = {
      if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) 1e3
      else 1e4
    } * (if (traitSet.containsIfApplicable(RelHomDistribution.SINGLE)) 1e2 else 1)
    val rowCount = mq.getRowCount(this)
    val bytesPerRow = getRowType.getFieldCount * 4
    planner.getCostFactory.makeCost(rowCount, rowCount * rf * bytesPerRow * 1e25, 0)
  }

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    val op = ("operator" , "pack")
    val child = getInput.asInstanceOf[PelagoRel].implement(target, alias)
    val childBinding = child._1
    val childOp = child._2

    val json = op ~ ("input", childOp)
    val ret = (childBinding, json)
    ret
  }

  override def getInputTraits: RelTraitSet = inTraits

  override def getTraitDef: RelTraitDef[_ <: RelTrait] = RelPackingTraitDef.INSTANCE
}

object PelagoPack {
  def create(input: RelNode, toPacking: RelPacking): PelagoPack = {
    val cluster = input.getCluster
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(toPacking)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
    new PelagoPack(input.getCluster, traitSet, input, toPacking)
  }
}

