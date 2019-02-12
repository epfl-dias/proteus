package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.emitter.Binding
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.`type`.RelDataType
import org.json4s.JsonDSL._
import org.json4s._
import org.apache.calcite.rex.RexNode
import org.json4s.JsonAST

import scala.collection.JavaConverters._
import java.util

import ch.epfl.dias.calcite.adapter.pelago.metadata.{PelagoRelMdDeviceType, PelagoRelMdDistribution, PelagoRelMetadataQuery}
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema, getFields}

/**
  * Implementation of {@link org.apache.calcite.rel.core.Project}
  * relational expression in Pelago.
  */
class PelagoProject protected (cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, projects: util.List[_ <: RexNode], rowType: RelDataType) //        assert getConvention() == input.getConvention();
  extends Project(cluster, traitSet, input, projects, rowType) with PelagoRel {
//  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, input: RelNode, projects: util.List[RexNode], rowType: RelDataType) = {
    PelagoProject.create(input, projects, rowType)
  }


  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    val rf = if (getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
      0.0001
    } else {
      0.01
    }
    val rf2 = if (getTraitSet.containsIfApplicable(RelHetDistribution.SINGLETON)) {
      1e5
    } else {
      1
    }
    val s = super.computeSelfCost(planner, mq)
    planner.getCostFactory.makeCost(
      s.getRows * rf2,
      s.getCpu * rf * rf2,
      s.getIo
    )
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  //almost 0 cost in Pelago
  override def implement(target: RelDeviceType): (Binding, JsonAST.JValue) = {
    val op      = ("operator" , "project")
    val alias   = "projection" + getId
    val rowType = emitSchema(alias, getRowType)
    val child   = getInput.asInstanceOf[PelagoRel].implement(target)
    val childBinding: Binding = child._1
    val childOp = child._2
    //TODO Could also use p.getNamedProjects
    val exprs = getNamedProjects
    val exprsJS: JValue = exprs.asScala.map {
      e => emitExpression(e.left,List(childBinding)).asInstanceOf[JsonAST.JObject] ~ ("register_as", ("attrName", e.right) ~ ("relName", alias))
    }

    val json = op ~
      ("gpu"          , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("relName"      , alias                                                 ) ~
      ("e"            , exprsJS                                               ) ~
      ("input"        , childOp          ) // ~ ("tupleType", rowType)
    val binding: Binding = Binding(alias,getFields(getRowType))
    val ret: (Binding, JValue) = (binding,json)
    ret
  }
}


object PelagoProject{
  def create(input: RelNode, projects: util.List[_ <: RexNode], rowType: RelDataType): PelagoProject = {
    val cluster  = input.getCluster
    val mq       = cluster.getMetadataQuery
    val dev      = PelagoRelMdDeviceType.project(mq, input, projects)
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION)
      .replaceIf(RelDistributionTraitDef.INSTANCE, () => PelagoRelMdDistribution.project(mq, input, projects))
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => dev);
    assert(traitSet.containsIfApplicable(RelPacking.UnPckd))
    new PelagoProject(cluster, traitSet, input, projects, rowType)
  }
}