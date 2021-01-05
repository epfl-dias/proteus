/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
 */

package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.metadata.{
  PelagoRelMdDeviceType,
  PelagoRelMetadataQuery
}
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits._
import ch.epfl.dias.emitter.PlanToJSON._
import ch.epfl.dias.emitter.{Binding, PlanConversionException}
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.{
  RelOptCluster,
  RelOptCost,
  RelOptPlanner,
  RelTraitSet
}
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.{Aggregate, AggregateCall}
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.ImmutableBitSet
import org.json4s.JsonDSL._
import org.json4s._

import java.util
import scala.collection.JavaConverters._

class PelagoAggregate protected (
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    hints: ImmutableList[RelHint],
    groupSet: ImmutableBitSet,
    groupSets: util.List[ImmutableBitSet],
    aggCalls: util.List[AggregateCall],
    var isGlobalAgg: Boolean,
    val isSplitted: Boolean
) extends Aggregate(
      cluster,
      traitSet,
      hints,
      input,
      groupSet,
      groupSets,
      aggCalls
    )
    with PelagoRel {

  override def copy(
      traitSet: RelTraitSet,
      input: RelNode,
      groupSet: ImmutableBitSet,
      groupSets: util.List[ImmutableBitSet],
      aggCalls: util.List[AggregateCall]
  ): PelagoAggregate = {
    PelagoAggregate.create(
      input,
      hints,
      groupSet,
      groupSets,
      aggCalls,
      isGlobalAgg,
      isSplitted
    )
  }

  def copy(
      input: RelNode,
      global: Boolean,
      isSplitted: Boolean
  ): PelagoAggregate = {
    PelagoAggregate.create(
      input,
      hints,
      groupSet,
      this.asInstanceOf[Aggregate].groupSets,
      aggCalls,
      global,
      isSplitted
    )
  }

  override def computeSelfCost(
      planner: RelOptPlanner,
      mq: RelMetadataQuery
  ): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(
      planner: RelOptPlanner,
      mq: RelMetadataQuery
  ): RelOptCost = {
    for (agg <- getAggCallList.asScala) {
      if (agg.getAggregation.getKind == SqlKind.AVG)
        return getCluster.getPlanner.getCostFactory.makeHugeCost()
    }

    val rf = {
      if (
        !isGlobalAgg && getTraitSet.containsIfApplicable(
          RelHomDistribution.RANDOM
        )
      ) 1e3
      else 1e5
    }

    val rf2 = {
      if (
        !isGlobalAgg && getTraitSet.containsIfApplicable(
          RelHetDistribution.SPLIT
        )
      ) 1e-15
      else 1e5
    }

    val base = super.computeSelfCost(planner, mq)

    val cpuCost =
      (if (
         !isGlobalAgg && getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)
       ) 1e-2
       else 1e9 * (1e-2 + 1e-1 * mq.getRowCount(getInput))) *
        rf * rf2 *
        Math.log(getInput.getRowType.getFieldCount + 10) *
        mq.getRowCount(getInput)

    planner.getCostFactory.makeCost(base.getRows, cpuCost, base.getIo)
  }

  protected lazy val rowEst: Long = {
    val exprow: Double =
      try {
        getCluster.getMetadataQuery.getRowCount(getInput)
      } catch {
        case _: Throwable => 1e20
      }
    Math.min(exprow, 256 * 1024 * 1024)
  }.asInstanceOf[Long]

  protected lazy val maxEst: Long = {
    val maxrow = getCluster.getMetadataQuery.getMaxRowCount(getInput)
    if (maxrow != null) Math.min(maxrow, 1024 * 1024 * 1024)
    else 1024 * 1024 * 1024
  }.asInstanceOf[Long]

  protected lazy val expectedKeyCardinality: Double = {
    val distinctRowCount = getCluster.getMetadataQuery.getDistinctRowCount(
      getInput,
      getGroupSet,
      null
    )
    if (distinctRowCount != null) {
      distinctRowCount
    } else {
      rowEst
    }
  }

  protected lazy val maxInputSize: Long = {
    Math.min(
      maxEst,
      (expectedKeyCardinality * 1.5 + {
        if (getTraitSet.contains(RelDeviceType.NVPTX)) 1024 * 1024 else 1024
      }).asInstanceOf[Long]
    )
  }

  protected lazy val hash_bits: Long = {
    Math.ceil(Math.log(expectedKeyCardinality) / Math.log(2)) + 1
  }.asInstanceOf[Long]

  override def explainTerms(pw: RelWriter): RelWriter = {
    super
      .explainTerms(pw)
      .item("trait", getTraitSet.toString)
      .item("global", isGlobalAgg)
      .item("hash_bits", hash_bits)
      .item("maxInputSize", maxInputSize)
  }

  override def implement(
      target: RelDeviceType,
      alias2: String
  ): (Binding, JValue) = {
    val alias = PelagoTable.create(alias2, getRowType)
    val op = ("operator", if (getGroupCount == 0) "reduce" else "groupby")
    val child = getInput.asInstanceOf[PelagoRel].implement(target)
    val childBinding: Binding = child._1
    val childOp = child._2

    val aggs: List[AggregateCall] = getAggCallList.asScala.toList

    val aggsJS = aggs.map {
      //        agg => emitAggExpression(agg, List(childBinding))
      agg => aggKind(agg.getAggregation)
    }

    val offset = getGroupCount

    val aggsExpr: List[JValue] = aggs.zipWithIndex.map {
      //        agg => emitAggExpression(agg, List(childBinding))
      agg =>
        {
          val arg = agg._1.getArgList
          val reg_as = (
            "attrName",
            getRowType.getFieldNames.get(agg._2 + offset)
          ) ~ ("relName", alias.getPelagoRelName)
          if (
            arg.size() == 1 && agg._1.getAggregation.getKind != SqlKind.COUNT
          ) {
            emitExpression(
              RexInputRef.of(arg.get(0), getInput.getRowType),
              List(childBinding),
              this
            ).asInstanceOf[JsonAST.JObject] ~ ("register_as", reg_as)
            //            emitArg(arg.get(0), List(childBinding)).asInstanceOf[JsonAST.JObject] ~ ("register_as", reg_as)
          } else if (
            arg.size() <= 1 && agg._1.getAggregation.getKind == SqlKind.COUNT
          ) {
            //FIXME: currently, count(A) ignores NULLs, so it is the same as count(*)
            ("expression", "int64") ~ ("v", 1) ~ ("register_as", reg_as)
          } else {
            //count() has 0 arguments; the rest expected to have 1
            val msg: String =
              "size of aggregate's input expected to be 0 or 1 - actually is " + arg
                .size()
            throw new PlanConversionException(msg)
          }
        }
    }

    if (getGroupCount == 0) {
      val json = op ~
        ("gpu", getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) ~
        ("e", aggsExpr) ~
        ("accumulator", aggsJS) ~
        ("input", childOp)
      val binding: Binding = Binding(alias, getFields(getRowType))
      val ret: (Binding, JValue) = (binding, json)
      ret
    } else {
      val aggK = getGroupSet.asScala.zipWithIndex.map(f => {
        val reg_as = (
          "attrName",
          getRowType.getFieldNames.get(f._2)
        ) ~ ("relName", alias.getPelagoRelName)
        emitExpression(
          RexInputRef.of(f._1, getInput.getRowType),
          List(childBinding),
          this
        ).asInstanceOf[JsonAST.JObject] ~ ("register_as", reg_as)
      })

      val aggE = aggsJS.zip(aggsExpr).zipWithIndex.map { z =>
        {
          ("m", z._1._1) ~
            ("e", z._1._2) ~
            ("packet", z._2 + 1) ~
            ("offset", 0) //FIXME: using different packets for each of them is the worst performance-wise
        }
      }

      val json = op ~
        ("gpu", getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) ~
        ("k", aggK) ~
        ("e", aggE) ~
        ("hash_bits", hash_bits) ~
        ("maxInputSize", maxInputSize.asInstanceOf[Long]) ~
        ("input", childOp)
      val binding: Binding = Binding(alias, getFields(getRowType))
      val ret: (Binding, JValue) = (binding, json)
      ret
    }
  }
}

object PelagoAggregate {
  def create(
      input: RelNode,
      hints: ImmutableList[RelHint],
      groupSet: ImmutableBitSet,
      groupSets: util.List[ImmutableBitSet],
      aggCalls: util.List[AggregateCall]
  ): PelagoAggregate = {
    create(
      input,
      hints,
      groupSet,
      groupSets,
      aggCalls,
      isGlobalAgg = true,
      isSplitted = false
    )
  }

  def create(
      input: RelNode,
      hints: ImmutableList[RelHint],
      groupSet: ImmutableBitSet,
      groupSets: util.List[ImmutableBitSet],
      aggCalls: util.List[AggregateCall],
      isGlobalAgg: Boolean,
      isSplitted: Boolean
  ): PelagoAggregate = {
    val cluster = input.getCluster
    val mq = cluster.getMetadataQuery
    val dev = PelagoRelMdDeviceType.aggregate(mq, input)
    val traitSet = input.getTraitSet
      .replace(PelagoRel.CONVENTION)
      .replace(PelagoRel.CONVENTION)
      .replaceIf(RelCollationTraitDef.INSTANCE, () => RelCollations.EMPTY)
      .replaceIf(
        RelHomDistributionTraitDef.INSTANCE,
        () => mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input)
      )
      .replaceIf(
        RelHetDistributionTraitDef.INSTANCE,
        () => mq.asInstanceOf[PelagoRelMetadataQuery].hetDistribution(input)
      )
      .replaceIf(
        RelComputeDeviceTraitDef.INSTANCE,
        () => RelComputeDevice.from(input)
      )
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => dev)
    new PelagoAggregate(
      cluster,
      traitSet,
      input,
      hints,
      groupSet,
      groupSets,
      aggCalls,
      isGlobalAgg,
      isSplitted
    )
  }
}
