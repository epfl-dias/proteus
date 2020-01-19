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
    extends SingleRel(cluster, traitSet, input) with PelagoRel with Converter {
  assert(getConvention eq PelagoRel.CONVENTION)
  assert(getConvention eq input.getConvention)
  protected var inTraits: RelTraitSet = input.getTraitSet

  override def copy(traitSet: RelTraitSet, inputs: java.util.List[RelNode]) = {
    assert(inputs.size() == 1)
    copy(traitSet, inputs.get(0))
  }

  def copy(traitSet: RelTraitSet, input: RelNode) = PelagoRouter.create(input, homdistribution)

  override def estimateRowCount(mq: RelMetadataQuery): Double = {
    var rc = mq.getRowCount(getInput)
    if      ((getHomDistribution eq RelHomDistribution.BRDCST) && (input.getTraitSet.contains(RelHomDistribution.RANDOM))) rc = rc * 2
    else if (getHomDistribution eq RelHomDistribution.BRDCST) rc = rc
    else if (getHomDistribution eq RelHomDistribution.RANDOM) rc = rc / 2.0 //TODO: Does this hold even when input is already distributed ?
    else if (getHomDistribution eq RelHomDistribution.SINGLE) rc = rc * 2.0 //TODO: Does this hold even when input is already distributed ?
    rc
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
//    if (traitSet.containsIfApplicable(RelPacking.UnPckd)) return planner.getCostFactory.makeInfiniteCost()
    val rf = 1
    val rf2 = 1 * (if (getTraitSet.containsIfApplicable(RelPacking.UnPckd)) 1e11 else 1)
    var base = super.computeSelfCost(planner, mq)
    //    if (getDistribution.getType eq RelDistribution.Type.HASH_DISTRIBUTED) base = base.multiplyBy(80)
    planner.getCostFactory.makeCost(base.getRows * rf2, base.getCpu * rf * rf2, base.getIo * rf2)
    //    planner.getCostFactory.makeZeroCost()
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  def getDistribution: RelDistribution = homdistribution.getDistribution

  def getHomDistribution: RelHomDistribution = homdistribution

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
//    assert(getTraitSet.containsIfApplicable(RelPacking.UnPckd) || (target != null))
    val child = getInput.asInstanceOf[PelagoRel].implement(target, alias)
    val childBinding: Binding = child._1
    var childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType, false, getTraitSet.containsIfApplicable(RelPacking.Packed))

    var out_dop = if (target == RelDeviceType.NVPTX) Repl.gpudop else Repl.cpudop
    if (homdistribution.satisfies(RelHomDistribution.SINGLE)) {
      out_dop = 1
    }

    var in_dop = if (input.getTraitSet.containsIfApplicable(RelComputeDevice.NVPTX) || input.getTraitSet.containsIfApplicable(RelComputeDevice.NVPTX)) Repl.gpudop else Repl.cpudop //FIXME: this is wrong... we should detect where it came from!
    if (input.getTraitSet.containsIfApplicable(RelHomDistribution.SINGLE)) {
      in_dop = 1
    }

    val projs = getRowType.getFieldList.asScala.zipWithIndex.map{
      f => {
        emitExpression(RexInputRef.of(f._2, getInput.getRowType), List(childBinding), this).asInstanceOf[JObject]
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
              ("relName", childBinding.rel.getPelagoRelName)
            ) ~
            ("attributes", List(
              ("relName", childBinding.rel.getPelagoRelName) ~
              ("attrName", "__broadcastTarget")
            ))
          ) ~
          ("attribute",
            ("relName", childBinding.rel.getPelagoRelName) ~
            ("attrName", "__broadcastTarget")
          )
        )
      } else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        if (getTraitSet.containsIfApplicable(RelPacking.Packed)) {
          JObject() //just leave the default policy
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

    var json = ("operator", "router") ~
      ("gpu"           , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("projections", rowType) ~
      ("numOfParents", out_dop) ~
      ("producers", in_dop) ~
      ("slack", 8) ~
      ("cpu_targets", target != RelDeviceType.NVPTX) ~
      policy ~
      ("input", childOp)

    if (getDistribution.getType != RelDistribution.Type.BROADCAST_DISTRIBUTED && getTraitSet.containsIfApplicable(RelPacking.Packed)) {
      json = ("operator", "mem-move-device") ~
        ("projections", emitSchema(childBinding.rel, getRowType, false, true)) ~
        ("input", json) ~
        ("to_cpu", target != RelDeviceType.NVPTX) ~
        ("do_transfer", getRowType.getFieldList.asScala.map(_ => true))
    }

    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }

  override def getInputTraits: RelTraitSet = inTraits

  override def getTraitDef: RelTraitDef[_ <: RelTrait] = RelDistributionTraitDef.INSTANCE
}

object PelagoRouter{
  def create(input: RelNode, distribution: RelHomDistribution): PelagoRouter = {
    assert(input.getTraitSet.contains(RelDeviceType.X86_64))
    val cluster  = input.getCluster
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input, false))
    new PelagoRouter(input.getCluster, traitSet, input, distribution)
  }
}
