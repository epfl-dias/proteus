package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.calcite.adapter.pelago.metadata.{PelagoRelMdDeviceType, PelagoRelMdDistribution, PelagoRelMetadataQuery}
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema}
import ch.epfl.dias.emitter.Binding
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan._
import org.apache.calcite.rel.{RelDistributionTraitDef, RelNode, RelWriter}
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.{RexNode, RexSimplify, RexUtil}
import org.json4s.JsonDSL._
import org.json4s._


class PelagoFilter protected (cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, condition: RexNode) extends Filter(cluster, traitSet, input, condition) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode) = PelagoFilter.create(input, condition)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    val rf = if (getTraitSet.containsIfApplicable(RelHetDistribution.SINGLETON)) 1e5 else 1
    if (getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) super.computeSelfCost(planner, mq).multiplyBy(0.001 * rf)
    else super.computeSelfCost(planner, mq).multiplyBy(10 * rf)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("isS", getTraitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX)).toString)

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    val op = ("operator" , "select")
    val child = getInput.asInstanceOf[PelagoRel].implement(target)
    val childBinding: Binding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)
    val cond = emitExpression(getCondition, List(childBinding))

    val json : JValue = op ~
      ("gpu"      , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("p"        , cond                                                  ) ~
      ("input"    , childOp                                               )

    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }
}

object PelagoFilter{
  def create(input: RelNode, condition: RexNode): PelagoFilter = {
    val cluster  = input.getCluster
    val mq       = cluster.getMetadataQuery
    val dev      = PelagoRelMdDeviceType.filter(mq, input)
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
      .replace(mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input))
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => dev);
    assert(traitSet.containsIfApplicable(RelPacking.UnPckd))
    new PelagoFilter(input.getCluster, traitSet, input, condition)
  }
}