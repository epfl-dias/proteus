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

package ch.epfl.dias.calcite.adapter.pelago.metadata

import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import com.google.common.collect.ImmutableRangeSet
import com.google.common.collect.Range
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.{
  BuiltInMetadata,
  ChainedRelMetadataProvider,
  MetadataDef,
  MetadataHandler,
  ReflectiveRelMetadataProvider,
  RelMdSelectivity,
  RelMdUtil,
  RelMetadataProvider,
  RelMetadataQuery
}
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rex._
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util._
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.fun.SqlStdOperatorTable

import scala.collection.JavaConverters._

object PelagoRelMdSelectivity {
  private val INSTANCE = new PelagoRelMdSelectivity
  val SOURCE: RelMetadataProvider = ChainedRelMetadataProvider.of(
    ImmutableList.of(
      ReflectiveRelMetadataProvider.reflectiveSource(
        BuiltInMethod.SELECTIVITY.method,
        PelagoRelMdSelectivity.INSTANCE
      ),
      RelMdSelectivity.SOURCE
    )
  )
//
//  public Double getSelectivity(PelagoPack rel, RelMetadataQuery mq,
//      RexNode predicate) {
//    return mq.getSelectivity(rel.getInput(), predicate);
//  }
//  public Double getSelectivity(PelagoUnpack rel, RelMetadataQuery mq,
//  public Double getSelectivity(PelagoRouter rel, RelMetadataQuery mq,
//    return rel.estimateRowCount(mq);//mq.getRowCount(rel.getInput()) / 2;
//  public Double getSelectivity(PelagoUnion rel, RelMetadataQuery mq,
//    return mq.getRowCount(rel.getInput(0)) + mq.getRowCount(rel.getInput(1));
//  public Double getSelectivity(PelagoDeviceCross rel, RelMetadataQuery mq,
//    return mq.getRowCount(rel.getInput());
  private def guessEqSelectivity(attr: RexTableInputRef): java.lang.Double = {
    val table = attr.getTableRef.getTable.unwrap(classOf[PelagoTable])
    val distinctValues =
      table.getDistrinctValues(ImmutableBitSet.of(attr.getIndex))
    if (distinctValues == null) return null
    1 / distinctValues
  }

  def getReferencedAttr(
      predicate: RexCall,
      mq: RelMetadataQuery,
      rel: RelNode
  ): RexTableInputRef = {
    val bi = predicate.getOperands
    assert(bi.size == 2)
// Check that exactly one of the inputs is a literal
    if (
      bi.get(0).isInstanceOf[RexLiteral] == bi.get(1).isInstanceOf[RexLiteral]
    ) return null
    val input =
      if (bi.get(1).isInstanceOf[RexLiteral]) bi.get(0)
      else bi.get(1)
    val `val` = (if (bi.get(1).isInstanceOf[RexLiteral]) bi.get(1)
                 else bi.get(0)).asInstanceOf[RexLiteral]
    val ls = mq.getExpressionLineage(rel, input)
    if (ls == null || ls.isEmpty) return null
// Check that it references a column, as we do not support transformations
    val a = ls.iterator.next
    if (!a.isInstanceOf[RexTableInputRef]) return null
    a.asInstanceOf[RexTableInputRef]
  }
  def getReferencedLiteral(
      predicate: RexCall,
      mq: RelMetadataQuery,
      rel: RelNode
  ): RexLiteral = {
    val bi = predicate.getOperands
    assert(bi.size == 2)
    if (
      bi.get(0).isInstanceOf[RexLiteral] == bi.get(1).isInstanceOf[RexLiteral]
    ) return null
    (if (bi.get(1).isInstanceOf[RexLiteral]) bi.get(1)
     else bi.get(0)).asInstanceOf[RexLiteral]
  }
  private class Estimator[C <: Comparable[C]] private[metadata] (
      val rexBuilder: RexBuilder,
      val `type`: RelDataType,
      val attr: RexTableInputRef,
      val ref: RexNode
  ) extends RangeSets.Consumer[C] {
    final private var table =
      attr.getTableRef.getTable.unwrap(classOf[PelagoTable])
    private var ret: Double = 0
    final private val remaining = ImmutableList.builder[RexNode]

    def getResult: Double = {
      val rems = remaining.build
      if (!rems.isEmpty)
        ret += RelMdUtil.guessSelectivity(
          RexUtil.composeConjunction(rexBuilder, rems)
        )
      ret
    }

    private def getLiteral(v: C) =
      rexBuilder.makeLiteral(v, `type`, false, false).asInstanceOf[RexLiteral]

    private def getPercentile(v: C) =
      table.getPercentile(
        ImmutableBitSet.of(attr.getIndex),
        getLiteral(v),
        rexBuilder
      )

    override def all(): Unit = ret += 1

    override def atLeast(lower: C): Unit = greaterThan(lower)

    override def atMost(upper: C): Unit = lessThan(upper)

    override def greaterThan(lower: C): Unit = {
      val local_sel = getPercentile(lower)
      if (local_sel == null)
        remaining.add(
          rexBuilder
            .makeCall(SqlStdOperatorTable.GREATER_THAN, ref, getLiteral(lower))
        )
      else ret += 1 - local_sel
    }

    override def lessThan(upper: C): Unit = {
      val local_sel = getPercentile(upper)
      if (local_sel == null)
        remaining.add(
          rexBuilder
            .makeCall(SqlStdOperatorTable.LESS_THAN, ref, getLiteral(upper))
        )
      else ret += local_sel
    }

    override def singleton(value: C): Unit = {
      val eq = guessEqSelectivity(attr)
      if (eq == null)
        remaining.add(
          rexBuilder
            .makeCall(SqlStdOperatorTable.EQUALS, ref, getLiteral(value))
        )
      else ret += eq
    }

    override def closed(lower: C, upper: C): Unit = {
      val up = getPercentile(upper)
      val dn = getPercentile(lower)
      if (up == null || dn == null)
        remaining.add(
          rexBuilder.makeCall(
            SqlStdOperatorTable.SEARCH,
            ref,
            rexBuilder.makeSearchArgumentLiteral(
              Sarg.of(false, ImmutableRangeSet.of(Range.closed(lower, upper))),
              ref.getType
            )
          )
        )
      else ret += getPercentile(upper) - getPercentile(lower)
    }

    override def closedOpen(lower: C, upper: C): Unit = closed(lower, upper)
    override def openClosed(lower: C, upper: C): Unit = closed(lower, upper)
    override def open(lower: C, upper: C): Unit = closed(lower, upper)
  }
}

class PelagoRelMdSelectivity protected ()
    extends MetadataHandler[BuiltInMetadata.Selectivity] {
  override def getDef: MetadataDef[BuiltInMetadata.Selectivity] =
    BuiltInMetadata.Selectivity.DEF

  def guessEqSelectivity(
      rel: RelNode,
      mq: RelMetadataQuery,
      predicate: RexNode
  ): java.lang.Double = {
    if (!predicate.isInstanceOf[RexCall]) return null
    if (!predicate.isA(SqlKind.EQUALS)) return null
    val bi = predicate.asInstanceOf[RexCall].getOperands
    assert(bi.size == 2)
    if (
      bi.get(0).isInstanceOf[RexLiteral] == bi.get(1).isInstanceOf[RexLiteral]
    ) return null
    val input =
      if (bi.get(1).isInstanceOf[RexLiteral]) bi.get(0)
      else bi.get(1)
    val ls = mq.getExpressionLineage(rel, input)
    if (ls == null || ls.isEmpty) return null
    val a = ls.iterator.next
    if (!a.isInstanceOf[RexTableInputRef]) return null
    PelagoRelMdSelectivity.guessEqSelectivity(a.asInstanceOf[RexTableInputRef])
  }

  def guessCmpSelectivity(
      rel: RelNode,
      mq: RelMetadataQuery,
      predicate: RexNode
  ): java.lang.Double = {
    if (!predicate.isInstanceOf[RexCall]) return null
    if (!predicate.isA(SqlKind.COMPARISON)) return null
    val attr = PelagoRelMdSelectivity.getReferencedAttr(
      predicate.asInstanceOf[RexCall],
      mq,
      rel
    )
    val lit = PelagoRelMdSelectivity.getReferencedLiteral(
      predicate.asInstanceOf[RexCall],
      mq,
      rel
    )
    if (attr == null) return null
    val table = attr.getTableRef.getTable.unwrap(classOf[PelagoTable])
    table.getPercentile(
      ImmutableBitSet.of(attr.getIndex),
      lit,
      rel.getCluster.getRexBuilder
    )
  }

  def guessBetweenSelectivity[C <: Comparable[C]](
      rel: RelNode,
      mq: RelMetadataQuery,
      predicate: RexNode
  ): java.lang.Double = {
    if (!predicate.isInstanceOf[RexCall]) return null
    if (!predicate.isA(SqlKind.SEARCH)) return null
    val call = predicate.asInstanceOf[RexCall]
    val attr = PelagoRelMdSelectivity.getReferencedAttr(
      predicate.asInstanceOf[RexCall],
      mq,
      rel
    )
    if (attr == null) return null
    val rexBuilder = rel.getCluster.getRexBuilder
    val ref = call.operands.get(0)
    val literal = call.operands.get(1).asInstanceOf[RexLiteral]
    val sarg =
      literal.getValueAs(classOf[Sarg[C]])
    var ret = 0.0
    if (sarg.containsNull)
      ret += 0.1 // TODO: ask statistics for percentage of null values
    if (sarg.isComplementedPoints) { // Generate 'ref <> value1 AND ... AND ref <> valueN'
// TODO: expand with more detailed statistics, as for now it assumes equal probability
//  for all values and all values to be part of the domain
      ret += 1 - sarg.rangeSet.asRanges.size * PelagoRelMdSelectivity
        .guessEqSelectivity(attr)
    } else {
      val consumer = new PelagoRelMdSelectivity.Estimator[C](
        rexBuilder,
        literal.getType,
        attr,
        ref
      )
      RangeSets.forEach(sarg.rangeSet, consumer)
      ret += consumer.getResult
    }
    Math.min(Math.max(ret, 0), 1)
  }

  def guessSelectivity(
      rel: RelNode,
      mq: RelMetadataQuery,
      predicate: RexNode
  ): java.lang.Double = {
    if (predicate == null) return null
    var sel = 1.0
    val remaining = ImmutableList.builder[RexNode]
    val bounds = new java.util.HashMap[RexTableInputRef, Pair[Boolean, Double]]

    for (pred <- RelOptUtil.conjunctions(predicate).asScala) {
      var local_sel: java.lang.Double = null
      if (pred.isA(SqlKind.EQUALS))
        local_sel = guessEqSelectivity(rel, mq, pred)
      else if (pred.isA(SqlKind.COMPARISON)) {
        local_sel = guessCmpSelectivity(rel, mq, pred)
        val lower_limit = pred.isA(SqlKind.GREATER_THAN) || pred.isA(
          SqlKind.GREATER_THAN_OR_EQUAL
        )
//        if (lower_limit) {
//          local_sel = 1 - local_sel;
//        }
        if (local_sel != null) {
          val ref = PelagoRelMdSelectivity.getReferencedAttr(
            pred.asInstanceOf[RexCall],
            mq,
            rel
          )
          if (bounds.containsKey(ref)) {
            val other = bounds.remove(ref)
            assert(other.left.booleanValue != lower_limit)
            local_sel = Math.abs(other.right - local_sel)
          } else {
            bounds.put(ref, new Pair[Boolean, Double](lower_limit, local_sel))
            local_sel = 1.0
          }
        }
      } else if (pred.isA(SqlKind.OR)) {
        local_sel = pred
          .asInstanceOf[RexCall]
          .getOperands
          .stream
          .map[java.lang.Double]((e: RexNode) => guessSelectivity(rel, mq, e))
          .reduce(
            0.0,
            (a: java.lang.Double, b: java.lang.Double) => a + b
          )
        local_sel = Math.min(Math.max(local_sel, 0), 1)
      } else if (pred.isA(SqlKind.SEARCH))
        local_sel = guessBetweenSelectivity(rel, mq, pred)
      if (local_sel == null) remaining.add(pred)
      else sel *= local_sel
    }

    for (v <- bounds.values.asScala) {
      var local_sel = v.right
      if (v.left) local_sel = 1 - local_sel
      sel *= local_sel
    }
    val rems = remaining.build
    if (!rems.isEmpty)
      sel *= RelMdUtil.guessSelectivity(
        RexUtil
          .composeConjunction(rel.getCluster.getRexBuilder, remaining.build)
      )
    sel
  }
  def getSelectivity(
      rel: RelNode,
      mq: RelMetadataQuery,
      predicate: RexNode
  ): Double = {
    val sel = guessSelectivity(rel, mq, predicate)
    if (sel != null) return sel
    RelMdUtil.guessSelectivity(predicate)
  }
}
