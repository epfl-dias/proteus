package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelComputeDevice, RelComputeDeviceTraitDef, RelDeviceType, RelDeviceTypeTraitDef, RelHomDistribution, RelPacking}
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.repl.Repl
import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet, _}
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, _}
import org.json4s.JsonDSL._
import org.json4s._

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
    if (traitSet.containsIfApplicable(RelPacking.UnPckd) && getHomDistribution == RelHomDistribution.BRDCST) return planner.getCostFactory.makeInfiniteCost()
    val rf = 1e7
    val inCnt = mq.getRowCount(input)
    val rf2 = 1 * (if (getTraitSet.containsIfApplicable(RelPacking.UnPckd)) 1e7 * (if (inCnt != null) inCnt.doubleValue() else 1e6) else 1)
    var base = super.computeSelfCost(planner, mq)
    //    if (getDistribution.getType eq RelDistribution.Type.HASH_DISTRIBUTED) base = base.multiplyBy(80)
    planner.getCostFactory.makeCost(base.getRows * rf2, base.getCpu * rf * rf2 * getRowType.getFieldCount, base.getIo * rf2)
    //    planner.getCostFactory.makeZeroCost()
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("policy", getHomDistribution)

  def getDistribution: RelDistribution = homdistribution.getDistribution

  def getHomDistribution: RelHomDistribution = homdistribution

  def hasMemMove: Boolean = {
    getTraitSet.containsIfApplicable(RelPacking.Packed)
  }

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
//    assert(getTraitSet.containsIfApplicable(RelPacking.UnPckd) || (target != null))
    val child = getInput.asInstanceOf[PelagoRel].implement(target, alias)
    val childBinding: Binding = child._1
    var childOp = child._2

    var out_dop = if (target == RelDeviceType.NVPTX) Repl.gpudop else Repl.cpudop
    if (homdistribution.satisfies(RelHomDistribution.SINGLE)) {
      out_dop = 1
    }

    var in_dop = if (input.getTraitSet.containsIfApplicable(RelComputeDevice.NVPTX) || input.getTraitSet.containsIfApplicable(RelComputeDevice.NVPTX)) Repl.gpudop else Repl.cpudop //FIXME: this is wrong... we should detect where it came from!
    if (input.getTraitSet.containsIfApplicable(RelHomDistribution.SINGLE)) {
      in_dop = 1
    }

    val policy: JObject = {
      if (getDistribution.getType eq RelDistribution.Type.BROADCAST_DISTRIBUTED) {
        if (getTraitSet.containsIfApplicable(RelPacking.Packed)) {
          childOp = ("operator", "mem-broadcast-device") ~
            ("num_of_targets", out_dop) ~
            ("input", child._2) ~
            ("to_cpu", target != RelDeviceType.NVPTX)
        }

        ("target",
          ("expression", "recordProjection") ~
          ("e",
            ("expression", "argument")
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
      ("numOfParents", out_dop) ~
      ("slack", 8) ~
      ("cpu_targets", target != RelDeviceType.NVPTX) ~
      policy ~
      ("input", childOp)

    if (getDistribution.getType != RelDistribution.Type.BROADCAST_DISTRIBUTED && getTraitSet.containsIfApplicable(RelPacking.Packed)) {
      json = ("operator", "mem-move-device") ~
        ("input", json) ~
        ("to_cpu", target != RelDeviceType.NVPTX) // ~
        // ("do_transfer", getRowType.getFieldList.asScala.map(_ => true))
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
