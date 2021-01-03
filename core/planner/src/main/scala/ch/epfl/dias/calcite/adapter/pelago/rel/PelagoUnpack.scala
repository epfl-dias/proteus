package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelComputeDevice, RelComputeDeviceTraitDef, RelDeviceType, RelDeviceTypeTraitDef, RelHetDistribution, RelHetDistributionTraitDef, RelHomDistribution, RelHomDistributionTraitDef, RelPacking, RelPackingTraitDef}

import java.util

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceTypeTraitDef
import ch.epfl.dias.emitter.Binding
import org.apache.calcite.plan._
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, SingleRel}
import org.json4s.JValue
import org.json4s.JsonDSL._

class PelagoUnpack protected(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode, val toPacking: RelPacking)
      extends SingleRel(cluster, traits, input) with PelagoRel with Converter {
  protected var inTraits: RelTraitSet = input.getTraitSet

  def getPacking() = toPacking;

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): PelagoUnpack = copy(traitSet, inputs.get(0), toPacking)

  def copy(traitSet: RelTraitSet, input: RelNode, packing: RelPacking) = PelagoUnpack.create(input, packing)


  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    ???
    // Higher cost if rows are wider discourages pushing a project through an
    val rf = {
      if (!getTraitSet.containsIfApplicable(RelHomDistribution.SINGLE)) {
//        if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) 0.0001
//        else
          0.001
//      } else if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
//        0.01
      } else {
        0.1
      }
    } * {
      if (!getTraitSet.containsIfApplicable(RelHetDistribution.SINGLETON)) {
        1
      } else {
        1e5
//        return planner.getCostFactory.makeHugeCost()
      }
    }
    // exchange.
    val rowCount = mq.getRowCount(this)
    val bytesPerRow = getRowType.getFieldCount * 4
    planner.getCostFactory.makeCost(rowCount, rowCount * bytesPerRow * rf * 100, rowCount * bytesPerRow)

//    if (input.getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == toDevice) planner.getCostFactory.makeHugeCost()
//    else planner.getCostFactory.makeTinyCost
  }

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    val op = ("operator", "unpack")
    val child = getInput.asInstanceOf[PelagoRel].implement(getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE), alias)
    val binding: Binding = child._1
    val childOp = child._2

    val json = op ~ ("input", childOp)

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
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
      .replaceIf(RelHomDistributionTraitDef.INSTANCE, () => mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input))
      .replaceIf(RelHetDistributionTraitDef.INSTANCE, () => mq.asInstanceOf[PelagoRelMetadataQuery].hetDistribution(input))
    new PelagoUnpack(input.getCluster, traitSet, input, toPacking)
  }
}

