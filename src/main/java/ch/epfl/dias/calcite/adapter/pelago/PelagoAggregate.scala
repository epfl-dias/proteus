package ch.epfl.dias.calcite.adapter.pelago

import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.{RelMdDeviceType, RelMdDistribution, RelMetadataQuery}
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.ImmutableBitSet
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import java.util

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMdDeviceType

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.{emitAggExpression, emitArg, emitSchema, emit_, formats, getFields}
import com.google.common.base.Supplier

import scala.collection.parallel.immutable

class PelagoAggregate protected(cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, indicator: Boolean,
                      groupSet: ImmutableBitSet, groupSets: util.List[ImmutableBitSet],
                      aggCalls: util.List[AggregateCall])
        extends Aggregate(cluster, traitSet, input, indicator, groupSet, groupSets, aggCalls) with PelagoRel {

  assert(getConvention eq PelagoRel.CONVENTION)
  assert(getConvention eq input.getConvention)

  override def copy(traitSet: RelTraitSet, input: RelNode, indicator: Boolean, groupSet: ImmutableBitSet,
                    groupSets: util.List[ImmutableBitSet], aggCalls: util.List[AggregateCall])
                          = {
    PelagoAggregate.create(input, indicator, groupSet, groupSets, aggCalls)
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    if (getTraitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX))) return planner.getCostFactory.makeTinyCost
    super.computeSelfCost(planner, mq).multiplyBy(0.1)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  override def implement: (Binding, JValue) = {
    val op = ("operator" , "agg")
    val child = getInput.asInstanceOf[PelagoRel].implement
    val childBinding: Binding = child._1
    val childOp = child._2

    val groups: List[Integer] = getGroupSet.toList.asScala.toList
    val groupsJS: JValue = groups.map {
      g => emitArg(g,List(childBinding))
    }

    val aggs: List[AggregateCall] = getAggCallList.asScala.toList
    val aggsJS = aggs.map {
      agg => emitAggExpression(agg,List(childBinding))
    }
    val alias = "agg" + getId
    val rowType = emitSchema(alias, getRowType)

    val json = op ~ ("tupleType", rowType) ~ ("groups", groupsJS) ~ ("aggs", aggsJS) ~ ("input" , childOp)
    val binding: Binding = Binding(alias,getFields(getRowType))
    val ret: (Binding, JValue) = (binding,json)
    ret
  }
}

object PelagoAggregate{
  def create(input: RelNode, indicator: Boolean, groupSet: ImmutableBitSet, groupSets: util.List[ImmutableBitSet], aggCalls: util.List[AggregateCall]): PelagoAggregate = {
    val cluster = input.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = cluster.traitSet
      .replace(PelagoRel.CONVENTION)
      .replace(RelDistributions.SINGLETON)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier[RelDeviceType]() {
        override def get: RelDeviceType = PelagoRelMdDeviceType.aggregate(mq, input)
      });
    new PelagoAggregate(cluster, traitSet, input, indicator, groupSet, groupSets, aggCalls)
  }
}