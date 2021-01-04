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
package ch.epfl.dias.calcite.adapter.pelago.rules

import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.rel._
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits._
import org.apache.calcite.plan.RelOptRule.{any, operand}
import org.apache.calcite.plan._
import org.apache.calcite.rel._
import org.apache.calcite.rel.convert.ConverterRule
import org.apache.calcite.rel.core._
import org.apache.calcite.rel.logical._
import org.apache.calcite.rel.rules.JoinCommuteRule
import org.apache.calcite.rex._
import org.apache.calcite.sql.SqlKind

import java.util

/**
  * Rules and relational operators for
  * {@link PelagoRel#CONVENTION}
  * calling convention.
  */
object PelagoRules {
  val RULES = Array(
    PelagoProjectTableScanRule.INSTANCE,
    PelagoToEnumerableConverterRule.INSTANCE,
    PelagoTableModifyRule.INSTANCE,
    PelagoProjectPushBelowUnpack.INSTANCE,
    PelagoProjectRule.INSTANCE,
    PelagoAggregateRule.INSTANCE,
    PelagoSortRule.INSTANCE,
    PelagoFilterRule.INSTANCE,
    PelagoUnnestRule.INSTANCE,
    PelagoScanRule.INSTANCE,
    PelagoValuesRule.INSTANCE, //        PelagoJoinSeq.INSTANCE,
    PelagoJoinSeq.INSTANCE2
  ) //Use the instance that swaps, as Lopt seems to generate left deep plans only

  /** Base class for planner rules that convert a relational expression to
    * Pelago calling convention. */

  abstract private[rules] class PelagoConverterRule[
      T <: RelNode
  ] private[rules] (
      val clazz: Class[T],
      description: String
  ) extends ConverterRule(
        clazz,
        new java.util.function.Predicate[T]() {
          override def test(t: T): Boolean = true
        },
        Convention.NONE,
        PelagoRel.CONVENTION,
        PelagoRelFactories.PELAGO_BUILDER,
        description
      ) {
    override def onMatch(call: RelOptRuleCall) = {
      val rel: RelNode = call.rel(0)
      if (rel.getTraitSet.contains(Convention.NONE)) {
        val converted = convert(rel)
        if (converted != null) call.transformTo(converted)
      }
    }
  }

  /**
    * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalProject}
    * to a {@link PelagoProject}.
    */
  object PelagoProjectRule { val INSTANCE = new PelagoRules.PelagoProjectRule }
  class PelagoProjectRule private ()
      extends PelagoRules.PelagoConverterRule(
        classOf[LogicalProject],
        "PelagoProjectRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def convert(rel: RelNode) = {
      val project = rel.asInstanceOf[Project]
      val traitSet = project.getInput.getTraitSet
        .replace(out)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
        .replace(RelHomDistribution.SINGLE)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
      PelagoProject.create(
        RelOptRule.convert(project.getInput, traitSet),
        project.getProjects,
        project.getRowType,
        project.getHints
      )
    }
  }
  object PelagoValuesRule { val INSTANCE = new PelagoRules.PelagoValuesRule }
  class PelagoValuesRule private ()
      extends PelagoRules.PelagoConverterRule(
        classOf[LogicalValues],
        "PelagoValuesRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def convert(rel: RelNode) = {
      val vals = rel.asInstanceOf[Values]
      val v =
        PelagoValues.create(vals.getCluster, vals.getRowType, vals.getTuples)
      rel.getCluster.getPlanner.ensureRegistered(v, rel)
      PelagoUnpack.create(v, RelPacking.UnPckd)
    }
  }
  object PelagoScanRule {
    val INSTANCE = new PelagoRules.PelagoScanRule

    /** Returns an array of integers {0, ..., n - 1}. */
    private def identityList(n: Int) = {
      val ints = new Array[Int](n)
      for (i <- 0 until n) { ints(i) = i }
      ints
    }
  }
  class PelagoScanRule private ()
      extends PelagoRules.PelagoConverterRule(
        classOf[LogicalTableScan],
        "PelagoScanRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def convert(rel: RelNode): RelNode = {
      val s = rel.asInstanceOf[LogicalTableScan]
      val relOptTable = s.getTable
      val pTable = relOptTable.unwrap(classOf[PelagoTable])
      if (pTable == null) return null
      val fieldCount = relOptTable.getRowType.getFieldCount
      val fields = PelagoScanRule.identityList(fieldCount)
      var scan: RelNode =
        PelagoTableScan.create(s.getCluster, relOptTable, pTable, fields)
      rel.getCluster.getPlanner.ensureRegistered(scan, s)
      if (pTable.getPacking eq RelPacking.Packed)
        scan = PelagoUnpack.create(scan, RelPacking.UnPckd)
      scan
    }
  }
  object PelagoTableModifyRule {
    val INSTANCE = new PelagoRules.PelagoTableModifyRule
  }
  class PelagoTableModifyRule private ()
      extends PelagoRules.PelagoConverterRule(
        classOf[LogicalTableModify],
        "PelagoTableModifyRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def convert(rel: RelNode) = {
      val mod = rel.asInstanceOf[LogicalTableModify]
      val traitSet = mod.getInput.getTraitSet
        .replace(out)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
        .replace(RelHomDistribution.SINGLE)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
        .replaceIf(RelCollationTraitDef.INSTANCE, () => RelCollations.EMPTY)
      PelagoTableModify.create(
        rel.getTable,
        mod.getCatalogReader,
        RelOptRule.convert(mod.getInput, traitSet),
        mod.getOperation,
        mod.getUpdateColumnList,
        mod.getSourceExpressionList,
        mod.isFlattened
      )
    }
  }

  /**
    * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalAggregate}
    * to a {@link PelagoProject}.
    */
  private object PelagoAggregateRule {
    val INSTANCE = new PelagoRules.PelagoAggregateRule
  }
  private class PelagoAggregateRule private ()
      extends PelagoRules.PelagoConverterRule(
        classOf[LogicalAggregate],
        "PelagoAggregateRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def convert(rel: RelNode) = {
      val agg = rel.asInstanceOf[Aggregate]
      val traitSet = agg.getTraitSet
        .replace(PelagoRel.CONVENTION)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
        .replace(RelHomDistribution.SINGLE)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
      val inp = RelOptRule.convert(agg.getInput, traitSet)
      PelagoAggregate.create(
        inp,
        agg.getHints,
        agg.getGroupSet,
        agg.getGroupSets,
        agg.getAggCallList,
        true,
        false
      )
    }
  }

  /**
    * Rule to create a {@link PelagoUnnest}.
    */
  private object PelagoUnnestRule {
    val INSTANCE = new PelagoRules.PelagoUnnestRule
  }
  private class PelagoUnnestRule private ()
      extends RelOptRule(
        operand(
          classOf[LogicalCorrelate],
          operand(classOf[RelNode], any),
          operand(classOf[Uncollect], operand(classOf[LogicalProject], any))
        ),
        PelagoRelFactories.PELAGO_BUILDER,
        "PelagoUnnestRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def onMatch(call: RelOptRuleCall): Unit = {
      val correlate = call.rel(0).asInstanceOf[LogicalCorrelate]
      if (!correlate.getTraitSet.contains(Convention.NONE)) return
      val input = call.rel(1)
      val uncollect = call.rel(2).asInstanceOf[Uncollect]
      val proj = call.rel(3).asInstanceOf[LogicalProject]
      val traitSet = uncollect.getTraitSet
        .replace(PelagoRel.CONVENTION)
        .replace(RelDeviceType.X86_64)
        .replace(RelHomDistribution.SINGLE)
        .replace(RelPacking.UnPckd)
      call.transformTo(
        PelagoUnnest.create(
          RelOptRule.convert(input, traitSet),
          correlate.getCorrelationId,
          proj.getNamedProjects,
          uncollect.getRowType
        )
      )
    }
  }

  /**
    * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalSort}
    * to a {@link PelagoSort}.
    */
  private object PelagoSortRule {
    val INSTANCE = new PelagoRules.PelagoSortRule
  }
  private class PelagoSortRule private ()
      extends PelagoRules.PelagoConverterRule(
        classOf[LogicalSort],
        "PelagoSortRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def convert(rel: RelNode) = {
      val sort = rel.asInstanceOf[Sort]
      val traitSet = sort.getInput.getTraitSet
        .replace(PelagoRel.CONVENTION)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
        .replace(RelHomDistribution.SINGLE)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
        .replaceIf(RelCollationTraitDef.INSTANCE, () => RelCollations.EMPTY)
      val inp = RelOptRule.convert(sort.getInput, traitSet)
      PelagoSort.create(inp, sort.collation, sort.offset, sort.fetch)
    }
  }

  /**
    * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalFilter} to a
    * {@link PelagoFilter}.
    */
  private object PelagoFilterRule {
    val INSTANCE = new PelagoRules.PelagoFilterRule
  }
  private class PelagoFilterRule private ()
      extends PelagoRules.PelagoConverterRule(
        classOf[LogicalFilter],
        "PelagoFilterRule"
      ) {
    override def matches(call: RelOptRuleCall) = true
    override def convert(rel: RelNode) = {
      val filter = rel.asInstanceOf[Filter]
      val traitSet = filter.getInput.getTraitSet
        .replace(out)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
        .replace(RelHomDistribution.SINGLE)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
      PelagoFilter.create(
        RelOptRule.convert(filter.getInput, traitSet),
        filter.getCondition
      )
    }
  }
  private object PelagoJoinSeq {
    private val INSTANCE =
      new PelagoRules.PelagoJoinSeq("PelagoJoinSeqRule", false)
    val INSTANCE2 = new PelagoRules.PelagoJoinSeq("PelagoJoinSeqRule2", true)
  }
  private class PelagoJoinSeq protected (description: String, val swap: Boolean)
      extends PelagoRules.PelagoConverterRule(
        classOf[PelagoLogicalJoin],
        description
      ) {

    final private val leftDeviceType = RelDeviceType.X86_64 //.NVPTX;

    final private val rightDeviceType = RelDeviceType.X86_64
    final private val leftDistribution = RelHomDistribution.SINGLE
    final private val rightDistribution = RelHomDistribution.SINGLE
    override def matches(call: RelOptRuleCall): Boolean = {
      val join = call.rel(0).asInstanceOf[Join]
//            if (join.getLeft().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE) != join.getRight().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)) return false;
      var condition = join.getCondition
      if (condition.isAlwaysTrue) return false
      val inf = join.analyzeCondition
      if (inf.isEqui) return true
      condition = RexUtil.toCnf(join.getCluster.getRexBuilder, condition)
//            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
//            if (disjunctions.size() != 1)  return false;
// Check that all conjunctions are equalities (only hashjoin supported)
//            condition = disjunctions.get(0);
      import scala.collection.JavaConversions._
      for (predicate <- RelOptUtil.conjunctions(condition)) {
        if (predicate.isA(SqlKind.EQUALS)) return true
      }
      false
    }
    override def convert(rel: RelNode) = {
      assert(false, "wrong convert called, as it needs RelOptRuleCall")
      null
    }
    override def onMatch(call: RelOptRuleCall) = {
      val rel = call.rel(0).asInstanceOf[RelNode]
      if (rel.getTraitSet.contains(Convention.NONE)) {
        val converted = convert(rel, call)
        if (converted != null) call.transformTo(converted)
      }
    }
    def convert(rel: RelNode, call: RelOptRuleCall): RelNode = {
      var join = rel.asInstanceOf[Join]
      val origJoin = join
      val cond = join.getCondition
      val inf = join.analyzeCondition
      val equalities = new util.ArrayList[RexNode]
      val rest = new util.ArrayList[RexNode]
      val rest0 = new util.ArrayList[RexNode]
      val rest1 = new util.ArrayList[RexNode]
      val thr = join.getLeft.getRowType.getFieldCount
      if (inf.isEqui) equalities.add(cond)
      else {
        val condition = RexUtil.pullFactors(join.getCluster.getRexBuilder, cond)
        assert(condition.isA(SqlKind.AND) || condition.isA(SqlKind.EQUALS))
        import scala.collection.JavaConversions._
        for (predicate <- condition.asInstanceOf[RexCall].getOperands) { // Needs a little bit of fixing... not completely correct checking
          val vis = new RelOptUtil.InputFinder
          predicate.accept(vis)
          var rel0 = false
          var rel1 = false
          import scala.collection.JavaConversions._
          for (acc <- RelOptUtil.InputFinder.bits(predicate)) {
            rel0 = rel0 || (acc < thr)
            rel1 = rel1 || (acc >= thr)
          }
          if (predicate.isA(SqlKind.EQUALS))
            if (rel0 && rel1) { equalities.add(predicate) }
            else if (rel0 && !rel1) rest0.add(predicate)
            else if (!rel0 && rel1) rest1.add(predicate)
            else rest.add(predicate)
        }
      }
      val rexBuilder = join.getCluster.getRexBuilder
      val joinCond = RexUtil.composeConjunction(rexBuilder, equalities, false)
      val leftCond = RexUtil.composeConjunction(rexBuilder, rest0, false)
      val rightCond = RexUtil
        .shift(RexUtil.composeConjunction(rexBuilder, rest1, false), -thr)
      val aboveCond = RexUtil.composeConjunction(rexBuilder, rest, false)
      val leftTraitSet = rel.getCluster.traitSet
        .replace(out)
        .replace(leftDistribution)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => leftDeviceType)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
      val rightTraitSet = rel.getCluster.traitSet
        .replace(out)
        .replace(rightDistribution)
        .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => rightDeviceType)
        .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.UnPckd)
      val preLeft = RelOptRule.convert(join.getLeft, leftTraitSet)
      val left =
        if (!(rest0.isEmpty))
          RelOptRule
            .convert(PelagoFilter.create(preLeft, leftCond), leftTraitSet)
        else preLeft
      val preRight = RelOptRule.convert(join.getRight, rightTraitSet)
      val right =
        if (!(rest1.isEmpty))
          RelOptRule
            .convert(PelagoFilter.create(preRight, rightCond), rightTraitSet)
        else preRight
      join = PelagoJoin
        .create(left, right, joinCond, join.getVariablesSet, join.getJoinType)
      var swapped =
        if (swap) JoinCommuteRule.swap(join, false, call.builder)
        else join
      if (swapped == null) return null
      if (swap) {
        val newJoin =
          if (swapped.isInstanceOf[Join]) swapped.asInstanceOf[Join]
          else swapped.getInput(0).asInstanceOf[Join]
        val relBuilder = call.builder
        val exps = RelOptUtil.createSwappedJoinExprs(newJoin, join, false)
        relBuilder.push(swapped).project(exps, newJoin.getRowType.getFieldNames)
        val build = relBuilder.build
        if (build ne newJoin)
          call.getPlanner.ensureRegistered(relBuilder.build, newJoin)
      }
      swapped = RelOptRule.convert(swapped, PelagoRel.CONVENTION)
      if (rest.isEmpty) return swapped
      val root = PelagoFilter.create(swapped, aboveCond)
      rel.getCluster.getPlanner.ensureRegistered(root, origJoin)
      root
    }
  }
}
class PelagoRules private () {}
