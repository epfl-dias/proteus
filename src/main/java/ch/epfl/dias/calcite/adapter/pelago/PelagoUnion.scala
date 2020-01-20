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
    if (!getInput(0).getTraitSet.containsIfApplicable(RelComputeDevice.X86_64)) return planner.getCostFactory.makeInfiniteCost()
    if (!getInput(1).getTraitSet.containsIfApplicable(RelComputeDevice.NVPTX )) return planner.getCostFactory.makeInfiniteCost()

    super.computeSelfCost(planner, mq).multiplyBy(1e8).plus(planner.getCostFactory.makeCost(0, 0, mq.getRowCount(this)))
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("isS", getTraitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX)).toString)

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    assert(all)
    val op = ("operator" , "union-all")
    val left_child = getInput(0).asInstanceOf[PelagoRel].implement(null, alias)
    val left_childBinding: Binding = left_child._1
    val left_childOp = left_child._2
    val right_child = getInput(1).asInstanceOf[PelagoRel].implement(null, alias)
    val right_childBinding: Binding = right_child._1
    val right_childOp = right_child._2
    val rowType = emitSchema(left_childBinding.rel, getRowType, false, getTraitSet.containsIfApplicable(RelPacking.Packed))

    val json : JValue = op ~
      ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("projections", rowType                                               ) ~
      ("input"      , List(left_childOp, right_childOp)                     )

    val ret: (Binding, JValue) = (left_childBinding, json)
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

