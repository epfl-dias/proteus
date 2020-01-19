package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.calcite.adapter.pelago.metadata.{PelagoRelMdDeviceType, PelagoRelMdDistribution, PelagoRelMetadataQuery}
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema}
import ch.epfl.dias.emitter.Binding
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan._
import org.apache.calcite.rel.{RelDistributionTraitDef, RelNode, RelWriter}
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.{RexBuilder, RexCall, RexNode, RexSimplify, RexUtil}
import org.json4s.JsonDSL._
import org.json4s._
import scala.collection.JavaConverters._

class PelagoFilter protected (cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, condition: RexNode) extends Filter(cluster, traitSet, input, condition) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode) = PelagoFilter.create(input, condition)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    var rf = if (getTraitSet.containsIfApplicable(RelHetDistribution.SINGLETON)) 1e0 else 1
//    val rf = 1
    if (getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
      if (getTraitSet.containsIfApplicable(RelHomDistribution.SINGLE)) rf = 1e10//return planner.getCostFactory.makeInfiniteCost()
      super.computeSelfCost(planner, mq).multiplyBy(0.001 * rf * 1e5)
    } else {
      if (getTraitSet.containsIfApplicable(RelHomDistribution.SINGLE)) rf = 1e10
      super.computeSelfCost(planner, mq).multiplyBy(10 * rf * 1e7)
    }
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("isS", getTraitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX)).toString)

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    val op = ("operator" , "select")
    val child = getInput.asInstanceOf[PelagoRel].implement(target, alias)
    val childBinding: Binding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)
    val cond = emitExpression(getCondition, List(childBinding), this)

    val json : JValue = op ~
      ("gpu"      , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("p"        , cond                                                  ) ~
      ("input"    , childOp                                               )

    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }
}

object PelagoFilter{
  def recFlatten(rexBuilder: RexBuilder, e: RexNode): RexNode ={
    if (!e.isInstanceOf[RexCall]) return e
    RexUtil.flatten(rexBuilder, e.asInstanceOf[RexCall].clone(e.getType, e.asInstanceOf[RexCall].getOperands.asScala.map(e => recFlatten(rexBuilder, e)).asJava))
  }


  def create(input: RelNode, condition: RexNode): PelagoFilter = {
    val cluster  = input.getCluster
    val mq       = cluster.getMetadataQuery
    val dev      = PelagoRelMdDeviceType.filter(mq, input)
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
      .replaceIf(RelHomDistributionTraitDef.INSTANCE, () => mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => dev);
    assert(traitSet.containsIfApplicable(RelPacking.UnPckd))
    new PelagoFilter(input.getCluster, traitSet, input, recFlatten(cluster.getRexBuilder, condition))
  }
}