package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.core.Exchange
import org.apache.calcite.rel.metadata.{RelMdDeviceType, RelMetadataQuery}
import org.json4s.JsonAST
import java.util

import ch.epfl.dias.emitter.PlanToJSON.{emitAggExpression, emitArg, emitExpression, emitSchema, emit_, getFields}
import ch.epfl.dias.emitter.Binding
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.{AggregateCall, Filter}
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode
import org.json4s.JsonAST._
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.apache.calcite.rex.RexNode
import org.json4s.JsonAST

import scala.collection.JavaConverters._
import java.util

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.{RelDeviceType, RelDeviceTypeTraitDef}
import com.google.common.base.Supplier
import org.apache.calcite.rel.convert.Converter

class PelagoRouter protected(cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, distribution: RelDistribution)
    extends Exchange(cluster, traitSet, input, distribution) with PelagoRel with Converter {
  assert(getConvention eq PelagoRel.CONVENTION)
  assert(getConvention eq input.getConvention)
  protected var inTraits: RelTraitSet = input.getTraitSet

  override def copy(traitSet: RelTraitSet, input: RelNode, distribution: RelDistribution) = PelagoRouter.create(input, distribution)

  override def estimateRowCount(mq: RelMetadataQuery): Double = {
    var rc = super.estimateRowCount(mq)
    if      (getDistribution eq RelDistributions.BROADCAST_DISTRIBUTED) rc = rc * 8.0
    else if (getDistribution eq RelDistributions.RANDOM_DISTRIBUTED   ) rc = rc / 8.0 //TODO: Does this hold even when input is already distributed ?
    rc
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    var base = super.computeSelfCost(planner, mq).multiplyBy(0.1)
//    if (getDistribution.getType eq RelDistribution.Type.HASH_DISTRIBUTED) base = base.multiplyBy(80)
    base
//    planner.getCostFactory.makeTinyCost()
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  override def implement: (Binding, JValue) = {
    val op = {
      if (getDistribution.getType eq RelDistribution.Type.BROADCAST_DISTRIBUTED) {
        ("operator", "broadcast")
      } else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        ("operator", "unionall")
      } else if (getDistribution.getType eq RelDistribution.Type.RANDOM_DISTRIBUTED) {
        ("operator", "split")
      } else {
        // else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        ("operator", "shuffle")
      }
    }

    val child = getInput.asInstanceOf[PelagoRel].implement
    val childBinding: Binding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)

    val json = op ~ ("tupleType", rowType) ~ ("input", childOp) ~ ("slack", 8)
    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }

  override def getInputTraits: RelTraitSet = inTraits

  override def getTraitDef: RelTraitDef[_ <: RelTrait] = RelDistributionTraitDef.INSTANCE
}

object PelagoRouter{
  def create(input: RelNode, distribution: RelDistribution): PelagoRouter = {
    val cluster  = input.getCluster
    val traitSet = cluster.traitSet.replace(PelagoRel.CONVENTION).replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier[RelDeviceType]() {
        override def get: RelDeviceType = {
//          System.out.println(RelMdDeviceType.exchange(cluster.getMetadataQuery, input) + " " + input.getTraitSet + " " + input)
          return RelMdDeviceType.exchange(cluster.getMetadataQuery, input)
        }
      });
    new PelagoRouter(input.getCluster, traitSet, input, distribution)
  }
}