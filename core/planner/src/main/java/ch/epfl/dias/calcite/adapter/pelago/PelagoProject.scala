package ch.epfl.dias.calcite.adapter.pelago

import java.util

import ch.epfl.dias.calcite.adapter.pelago.metadata.{PelagoRelMdDeviceType, PelagoRelMdHomDistribution}
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema, getFields}
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel._
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.{RelMdCollation, RelMetadataQuery}
import org.apache.calcite.rex.RexNode
import org.json4s.JsonDSL._
import org.json4s.{JsonAST, _}

import scala.collection.JavaConverters._

/**
  * Implementation of {@link org.apache.calcite.rel.core.Project}
  * relational expression in Pelago.
  */
class PelagoProject protected (cluster: RelOptCluster, traitSet: RelTraitSet, hints: ImmutableList[RelHint],
                               input: RelNode, projects: util.List[_ <: RexNode], rowType: RelDataType) //        assert getConvention() == input.getConvention();
  extends Project(cluster, traitSet, hints, input, projects, rowType) with PelagoRel {
//  assert(getConvention eq PelagoRel.CONVENTION())

  override def copy(traitSet: RelTraitSet, input: RelNode, projects: util.List[RexNode], rowType: RelDataType) = {
    PelagoProject.create(input, projects, rowType, hints)
  }


  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    // The project is 0-cost in proteus, especially for trivial projections
    // Nevertheless, if we do not put a big multiplicative factor, it's cost
    // is negligible compared to the rest of the plan and thus the optimizer
    // two consecutive projects in favor of a single one, due to their cost
    // equivalence. This increases the search space as there are have more
    // intermediate results. We use this big factor to compensate for that.
    val c = super.computeSelfCost(planner, mq)
    planner.getCostFactory.makeCost(c.getRows, c.getCpu * 1e15, c.getIo)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  //almost 0 cost in Pelago
  override def implement(target: RelDeviceType, alias2: String): (Binding, JsonAST.JValue) = {
    val alias   = PelagoTable.create(alias2, getRowType)
    val op      = ("operator" , "project")
    val rowType = emitSchema(alias, getRowType)
    val child   = getInput.asInstanceOf[PelagoRel].implement(target, alias2)
    val childBinding: Binding = child._1
    val childOp = child._2
    //TODO Could also use p.getNamedProjects
    val exprs = getNamedProjects
    val exprsJS: JValue = exprs.asScala.map {
      e => emitExpression(e.left,List(childBinding), this).asInstanceOf[JsonAST.JObject] ~ ("register_as", ("attrName", e.right) ~ ("relName", alias.getPelagoRelName))
    }

    val json = op ~
      ("relName"      , alias.getPelagoRelName                                ) ~
      ("e"            , exprsJS                                               ) ~
      ("input"        , childOp          ) // ~ ("tupleType", rowType)
    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding,json)
    ret
  }
}


object PelagoProject{
  def create(input: RelNode, projects: util.List[_ <: RexNode], rowType: RelDataType, hints: ImmutableList[RelHint]): PelagoProject = {
    val cluster  = input.getCluster
    val mq       = cluster.getMetadataQuery
    val dev      = PelagoRelMdDeviceType.project(mq, input, projects)
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION)
      .replaceIfs(RelCollationTraitDef.INSTANCE, () => RelMdCollation.project(mq, input, projects))
      .replaceIf(RelHomDistributionTraitDef.INSTANCE, () => PelagoRelMdHomDistribution.project(mq, input, projects))
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => dev);
    assert(traitSet.containsIfApplicable(RelPacking.UnPckd))
    new PelagoProject(cluster, traitSet, hints, input, projects, rowType)
  }
}