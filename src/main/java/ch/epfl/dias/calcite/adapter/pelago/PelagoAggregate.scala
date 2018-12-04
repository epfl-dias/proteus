package ch.epfl.dias.calcite.adapter.pelago

import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.{RelMdDistribution, RelMetadataQuery}
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rex.{RexInputRef, RexNode}
import org.apache.calcite.util.ImmutableBitSet
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import java.util

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMdDeviceType
import ch.epfl.dias.emitter.PlanConversionException
import org.apache.calcite.sql.{SqlAggFunction, SqlKind}

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import com.google.common.base.Supplier

import scala.collection.parallel.immutable

class PelagoAggregate protected(cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, indicator: Boolean,
                      groupSet: ImmutableBitSet, groupSets: util.List[ImmutableBitSet],
                      aggCalls: util.List[AggregateCall],
                      var isGlobalAgg: Boolean)
        extends Aggregate(cluster, traitSet, input, indicator, groupSet, groupSets, aggCalls) with PelagoRel {

  override def copy(traitSet: RelTraitSet, input: RelNode, indicator: Boolean, groupSet: ImmutableBitSet,
                    groupSets: util.List[ImmutableBitSet], aggCalls: util.List[AggregateCall])
                          = {
    PelagoAggregate.create(input, indicator, groupSet, groupSets, aggCalls, isGlobalAgg)
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    for (agg <- getAggCallList().asScala){
        if (agg.getAggregation().getKind() == SqlKind.AVG) return getCluster.getPlanner.getCostFactory.makeHugeCost();
    }

    val rf = {
      if (getTraitSet.containsIfApplicable(RelDistributions.RANDOM_DISTRIBUTED)) 1e-3
      else 1e2
    }


    if (getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
      super.computeSelfCost(planner, mq).multiplyBy(0.1 * 1e2 * rf * Math.log(getInput.getRowType.getFieldCount))
    } else {
      super.computeSelfCost(planner, mq).multiplyBy(10 * 1e2 * rf * Math.log(getInput.getRowType.getFieldCount))
    }
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    val op = ("operator", if (getGroupCount == 0) "reduce" else "groupby")
    val child = getInput.asInstanceOf[PelagoRel].implement(target)
    val childBinding: Binding = child._1
    val childOp = child._2

    val alias = "agg" + getId

    val aggs: List[AggregateCall] = getAggCallList.asScala.toList

    val aggsJS = aggs.map {
      //        agg => emitAggExpression(agg, List(childBinding))
      agg => aggKind(agg.getAggregation)
    }

    val offset = getGroupCount

    val aggsExpr = aggs.zipWithIndex.map {
      //        agg => emitAggExpression(agg, List(childBinding))
      agg => {
        val arg    = agg._1.getArgList
        val reg_as = ("attrName", getRowType.getFieldNames.get(agg._2 + offset)) ~ ("relName", alias)
        if (arg.size() == 1 && agg._1.getAggregation.getKind != SqlKind.COUNT) {
          emitExpression(RexInputRef.of(arg.get(0), getInput.getRowType), List(childBinding)).asInstanceOf[JsonAST.JObject] ~ ("register_as", reg_as)
          //            emitArg(arg.get(0), List(childBinding)).asInstanceOf[JsonAST.JObject] ~ ("register_as", reg_as)
        } else if (arg.size() <= 1 && agg._1.getAggregation.getKind == SqlKind.COUNT) {
          //FIXME: currently, count(A) ignores NULLs, so it is the same as count(*)
          ("expression", "int64") ~ ("v", 1) ~ ("register_as", reg_as)
        } else {
          //count() has 0 arguments; the rest expected to have 1
          val msg : String = "size of aggregate's input expected to be 0 or 1 - actually is " + arg.size()
          throw new PlanConversionException(msg)
        }
      }
    }

    if (getGroupCount == 0) {
      val groups: List[Integer] = getGroupSet.toList.asScala.toList
      val groupsJS: JValue = groups.map {
        g => emitArg(g, List(childBinding))
      }

      val rowType = emitSchema(alias, getRowType)

      val json = op ~
        ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
        ("e"          , aggsExpr                                              ) ~
        ("accumulator", aggsJS                                                ) ~
        ("p"          , ("expression", "bool") ~ ("v", true)                  ) ~
        ("input"      , childOp)
      val binding: Binding = Binding(alias, getFields(getRowType))
      val ret: (Binding, JValue) = (binding, json)
      ret
    } else {
      val aggK = getGroupSet.asScala.zipWithIndex.map(f => {
        val reg_as = ("attrName", getRowType.getFieldNames.get(f._2)) ~ ("relName", alias)
        emitExpression(RexInputRef.of(f._1, getInput.getRowType), List(childBinding)).asInstanceOf[JsonAST.JObject] ~ ("register_as", reg_as)
      })

      val aggE =  aggsJS.zip(aggsExpr).zipWithIndex.map {
        z => {
          ("m"     , z._1._1 ) ~
          ("e"     , z._1._2 ) ~
          ("packet", z._2 + 1) ~
          ("offset", 0       )        //FIXME: using different packets for each of them is the worst performance-wise
        }
      }

      //FIXME: reconsider these upper limits
      val rowEst = Math.min(getInput.estimateRowCount(getCluster.getMetadataQuery), 64*1024*1024) //1 vs 128 vs 64
      val maxrow = getCluster.getMetadataQuery.getMaxRowCount(getInput  )
      val maxEst = if (maxrow != null) Math.min(maxrow, 64*1024*1024) else 64*1024*1024 //1 vs 128 vs 64

      val hash_bits = Math.min(1 + Math.ceil(Math.log(rowEst)/Math.log(2)).asInstanceOf[Int], 15)

      val json = op ~
        ("gpu"          , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
        ("k"            , aggK                                                  ) ~
        ("e"            , aggE                                                  ) ~
        ("hash_bits"    , hash_bits                                             ) ~
        ("maxInputSize" , maxEst.asInstanceOf[Int]                              ) ~
        ("input"        , childOp                                               )
      val binding: Binding = Binding(alias, getFields(getRowType))
      val ret: (Binding, JValue) = (binding, json)
      ret
    }
  }
}

object PelagoAggregate{
  def create(input: RelNode, indicator: Boolean, groupSet: ImmutableBitSet, groupSets: util.List[ImmutableBitSet], aggCalls: util.List[AggregateCall]): PelagoAggregate = {
    create(input, indicator, groupSet, groupSets, aggCalls, false)
  }

  def create(input: RelNode, indicator: Boolean, groupSet: ImmutableBitSet, groupSets: util.List[ImmutableBitSet], aggCalls: util.List[AggregateCall], isGlobalAgg: Boolean): PelagoAggregate = {
    val cluster = input.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = cluster.traitSet
      .replace(PelagoRel.CONVENTION)
//      .replace(RelDistributions.SINGLETON)
      .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier[RelDistribution]() {
        override def get: RelDistribution = mq.distribution(input);
      })
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier[RelDeviceType]() {
        override def get: RelDeviceType = PelagoRelMdDeviceType.aggregate(mq, input)
      });
    new PelagoAggregate(cluster, traitSet, input, indicator, groupSet, groupSets, aggCalls, isGlobalAgg)
  }
}