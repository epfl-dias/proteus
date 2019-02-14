package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.Exchange
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema}
import ch.epfl.dias.emitter.Binding
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexInputRef
import org.json4s.JsonDSL._
import org.json4s._

import scala.collection.JavaConverters._
import ch.epfl.dias.repl.Repl
import org.apache.calcite.rel.convert.Converter

class PelagoRouter protected(cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, val homdistribution: RelHomDistribution)
    extends Exchange(cluster, traitSet, input, homdistribution.getDistribution()) with PelagoRel with Converter {
  assert(getConvention eq PelagoRel.CONVENTION)
  assert(getConvention eq input.getConvention)
  protected var inTraits: RelTraitSet = input.getTraitSet

  override def copy(traitSet: RelTraitSet, input: RelNode, distribution: RelDistribution) = PelagoRouter.create(input, RelHomDistribution.from(distribution))

  override def estimateRowCount(mq: RelMetadataQuery): Double = {
    val rc = mq.getRowCount(getInput)
//    if      (getDistribution eq RelDistributions.BROADCAST_DISTRIBUTED) rc = rc * 4.0
//    else if (getDistribution eq RelDistributions.RANDOM_DISTRIBUTED   ) rc = rc / 4.0 //TODO: Does this hold even when input is already distributed ?
    rc
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    //    if (traitSet.containsIfApplicable(RelPacking.UnPckd)) return planner.getCostFactory.makeHugeCost()
    val rf = 1e-6 * (if (distribution == RelHomDistribution.BRDCST) 10 else 1)
    val rf2 = 1e-6 * (if (getTraitSet.containsIfApplicable(RelPacking.UnPckd)) 1e6 else 1)
    var base = super.computeSelfCost(planner, mq)
    //    if (getDistribution.getType eq RelDistribution.Type.HASH_DISTRIBUTED) base = base.multiplyBy(80)
    planner.getCostFactory.makeCost(base.getRows, base.getCpu * rf * rf2, base.getIo)
    //    planner.getCostFactory.makeZeroCost()
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  override def getDistribution: RelDistribution = homdistribution.getDistribution

  def getHomDistribution: RelHomDistribution = homdistribution

  override def implement(target: RelDeviceType): (Binding, JValue) = {
//    assert(getTraitSet.containsIfApplicable(RelPacking.UnPckd) || (target != null))
    val child = getInput.asInstanceOf[PelagoRel].implement(null)
    val childBinding: Binding = child._1
    var childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType, false, getTraitSet.containsIfApplicable(RelPacking.Packed))

    var out_dop = Repl.gpudop // if (target == RelDeviceType.NVPTX) Repl.gpudop else Repl.cpudop
    if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
      out_dop = 1
    }

    var in_dop = Repl.gpudop // if (target == RelDeviceType.NVPTX) Repl.gpudop else Repl.cpudop //FIXME: this is wrong... we should detect where it came from!
    if (input.getTraitSet.containsIfApplicable(RelDistributions.SINGLETON)) {
      in_dop = 1
    }

    val op = {
      if (getDistribution.getType eq RelDistribution.Type.BROADCAST_DISTRIBUTED) {
        ("operator", "broadcast")
      } else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        ("operator", "unionall")
      } else if (getDistribution.getType eq RelDistribution.Type.RANDOM_DISTRIBUTED) {
        ("operator", "split")
      } else {
        ("operator", "shuffle")
//        assert(false, "translation not implemented")
      }
    }

    val projs = getRowType.getFieldList.asScala.zipWithIndex.map{
      f => {
        emitExpression(RexInputRef.of(f._2, getInput.getRowType), List(childBinding)).asInstanceOf[JObject]
      }
    }

    val policy: JObject = {
      if (getDistribution.getType eq RelDistribution.Type.BROADCAST_DISTRIBUTED) {
        if (getTraitSet.containsIfApplicable(RelPacking.Packed)) {
          childOp = ("operator", "mem-broadcast-device") ~
            ("num_of_targets", out_dop) ~
            ("projections", emitSchema(childBinding.rel, getRowType, false, true)) ~
            ("input", child._2) ~
            ("to_cpu", target != RelDeviceType.NVPTX)
        }

        ("target",
          ("expression", "recordProjection") ~
          ("e",
            ("expression", "argument") ~
            ("argNo", -1) ~
            ("type",
              ("type", "record") ~
              ("relName", childBinding.rel)
            ) ~
            ("attributes", List(
              ("relName", childBinding.rel) ~
              ("attrName", "__broadcastTarget")
            ))
          ) ~
          ("attribute",
            ("relName", childBinding.rel) ~
            ("attrName", "__broadcastTarget")
          )
        )
      } else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        if (getTraitSet.containsIfApplicable(RelPacking.Packed)) {
          ("operator", "exchange") //just leave the default policy
        } else {
          ("numa_local", false)
        }
      } else if (getDistribution.getType eq RelDistribution.Type.RANDOM_DISTRIBUTED) {
        if (getTraitSet.containsIfApplicable(RelPacking.Packed)) {
          if (target == RelDeviceType.NVPTX) {
            ("numa_local", true)
          } else {
            ("rand_local_cpu", true)
          }
        } else {
          ("numa_local", false)
        }
      } else {
        // else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        assert(false, "translation not implemented")
        ("operator", "shuffle")
      }
    }

    var json = ("operator", "exchange") ~
      ("gpu"           , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("projections", rowType) ~
      ("numOfParents", out_dop) ~
      ("producers", in_dop) ~
      ("slack", 8) ~
      policy ~
      ("input", childOp)

    if (getDistribution.getType != RelDistribution.Type.BROADCAST_DISTRIBUTED && getTraitSet.containsIfApplicable(RelPacking.Packed)) {
      json = ("operator", "mem-move-device") ~
        ("projections", emitSchema(childBinding.rel, getRowType, false, true)) ~
        ("input", json) ~
        ("to_cpu", target != RelDeviceType.NVPTX)
    }

    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }

  override def getInputTraits: RelTraitSet = inTraits

  override def getTraitDef: RelTraitDef[_ <: RelTrait] = RelDistributionTraitDef.INSTANCE
}

object PelagoRouter{
  def create(input: RelNode, distribution: RelHomDistribution): PelagoRouter = {
    val cluster  = input.getCluster
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input, false))
    new PelagoRouter(input.getCluster, traitSet, input, distribution)
  }
}
