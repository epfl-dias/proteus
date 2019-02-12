package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema}
import org.apache.calcite.plan._
import org.apache.calcite.rel.core.{Filter, Union}
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, RelWriter}
import org.apache.calcite.rex.RexNode
import org.json4s.JsonDSL._
import org.json4s._


class PelagoUnion protected
      (cluster: RelOptCluster, traitSet: RelTraitSet, inputs: java.util.List[RelNode], all: Boolean)
    extends Union(cluster, traitSet, inputs, all) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)
  assert(all)

  override def copy(traitSet: RelTraitSet, inputs: java.util.List[RelNode], all: Boolean) = PelagoUnion.create(inputs, all)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    if (getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) super.computeSelfCost(planner, mq).multiplyBy(0.001)
    else super.computeSelfCost(planner, mq).multiplyBy(10)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("isS", getTraitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX)).toString)

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    assert(all)
    val op = ("operator" , "union-all")
    val child = getInput(0).asInstanceOf[PelagoRel].implement(target)
    val childBinding: Binding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)
//    val cond = emitExpression(getCondition, List(childBinding))

    val json : JValue = op ~
      ("gpu"      , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("input"    , childOp                                               )

    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }
}

object PelagoUnion{
  def create(inputs: java.util.List[RelNode], all: Boolean): PelagoUnion = {
    val traitSet = inputs.get(0).getTraitSet.replace(PelagoRel.CONVENTION)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.NONE)
      .replaceIf(RelHetDistributionTraitDef.INSTANCE, () => RelHetDistribution.SINGLETON)
    new PelagoUnion(inputs.get(0).getCluster, traitSet, inputs, all)
  }
}

