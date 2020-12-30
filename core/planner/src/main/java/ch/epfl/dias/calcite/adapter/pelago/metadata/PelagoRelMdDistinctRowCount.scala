package ch.epfl.dias.calcite.adapter.pelago.metadata

import ch.epfl.dias.calcite.adapter.pelago._
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMdUtil.numDistinctVals
import org.apache.calcite.rel.metadata._
import org.apache.calcite.rex._
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`type`.{BasicSqlType, SqlTypeName}
import org.apache.calcite.util.{BuiltInMethod, ImmutableBitSet, NumberUtil}

import java.util
import scala.collection.JavaConverters._

object PelagoRelMdDistinctRowCount {
  private val INSTANCE = new PelagoRelMdDistinctRowCount
  val SOURCE: RelMetadataProvider = ChainedRelMetadataProvider.of(ImmutableList.of(
    ReflectiveRelMetadataProvider.reflectiveSource(
      BuiltInMethod.DISTINCT_ROW_COUNT.method,
      PelagoRelMdDistinctRowCount.INSTANCE
    ), RelMdDistinctRowCount.SOURCE))
}

class PelagoRelMdDistinctRowCount private() extends MetadataHandler[BuiltInMetadata.DistinctRowCount] {
  override def getDef: MetadataDef[BuiltInMetadata.DistinctRowCount] = BuiltInMetadata.DistinctRowCount.DEF

  def getDistinctRowCount(rel: PelagoUnpack, mq: RelMetadataQuery,
                          groupKey: ImmutableBitSet, predicate: RexNode): java.lang.Double = mq.getDistinctRowCount(rel.getInput, groupKey, predicate)

  def getDistinctRowCount(rel: PelagoPack, mq: RelMetadataQuery,
                          groupKey: ImmutableBitSet, predicate: RexNode): java.lang.Double = mq.getDistinctRowCount(rel.getInput, groupKey, predicate)

  def getDistinctRowCount(rel: PelagoRouter, mq: RelMetadataQuery,
                          groupKey: ImmutableBitSet, predicate: RexNode): java.lang.Double = mq.getDistinctRowCount(rel.getInput, groupKey, predicate)

  def getDistinctRowCount(rel: PelagoDeviceCross, mq: RelMetadataQuery,
                          groupKey: ImmutableBitSet, predicate: RexNode): java.lang.Double = mq.getDistinctRowCount(rel.getInput, groupKey, predicate)

  /** Visitor that walks over a scalar expression and computes the
   * cardinality of its result. */
  private class CardOfProjExpr private[metadata](val mq: RelMetadataQuery, var rel: Project) extends RexVisitorImpl[java.lang.Double](true) {
    override def visitInputRef(`var`: RexInputRef): java.lang.Double = {
      val index: Int = `var`.getIndex
      val col: ImmutableBitSet = ImmutableBitSet.of(index)
      val distinctRowCount = mq.getDistinctRowCount(rel.getInput, col, null)

      if (distinctRowCount == null) {
        return null
      }
      else {
        return RelMdUtil.numDistinctVals(distinctRowCount, mq.getRowCount(rel))
      }
    }

    override def visitLiteral(literal: RexLiteral): java.lang.Double = {
      return numDistinctVals(1.0, mq.getRowCount(rel))
    }

    override def visitCall(call: RexCall): java.lang.Double = {
      var distinctRowCount: Double = .0
      val rowCount: Double = mq.getRowCount(rel)
      if (call.isA(SqlKind.MINUS_PREFIX)) {
        distinctRowCount = cardOfProjExpr(mq, rel, call.getOperands.get(0))
      }
      else {
        if (call.isA(ImmutableList.of(SqlKind.PLUS, SqlKind.MINUS))) {
          val card0 = cardOfProjExpr(mq, rel, call.getOperands.get(0))
          if (card0 == null) {
            return null
          }
          val card1 = cardOfProjExpr(mq, rel, call.getOperands.get(1))
          if (card1 == null) {
            return null
          }
          distinctRowCount = Math.max(card0, card1)
        }
        else {
          if (call.isA(ImmutableList.of(SqlKind.TIMES, SqlKind.DIVIDE))) {
            val x = cardOfProjExpr(mq, rel, call.getOperands.get(0))
            val y = cardOfProjExpr(mq, rel, call.getOperands.get(1))
            distinctRowCount = NumberUtil.multiply(x, y)
            try {
              if (call.isA(SqlKind.DIVIDE) && call.getOperands.get(1).isA(SqlKind.LITERAL) && call.getOperands.get(0).getType.isInstanceOf[BasicSqlType]) {
                val lit = call.getOperands.get(1).asInstanceOf[RexLiteral]

                var overflow = call.getOperands.get(0).getType.asInstanceOf[BasicSqlType].getLimit(/* upper limit */ true, SqlTypeName.Limit.OVERFLOW, true).asInstanceOf[java.math.BigDecimal];
                // let's overestimate by two for the sign
                overflow = overflow.multiply(new java.math.BigDecimal(2))


                val lin = mq.getExpressionLineage(rel.getInput, call.getOperands.get(0))
                if (lin.size() == 1) {
                  try {
                    val tbl = lin.asScala.head.asInstanceOf[RexTableInputRef].getTableRef.getTable.unwrap(classOf[PelagoTable])
                    val ind = lin.asScala.head.asInstanceOf[RexTableInputRef].getIndex
                    overflow = overflow.min(RexLiteral.fromJdbcString(lit.getType, lit.getTypeName,
                      tbl.getRangeValues(ImmutableBitSet.of(ind)).right.asInstanceOf[String]).getValue3.asInstanceOf[java.math.BigDecimal])
                  } catch {
                    case _: Throwable => System.err.println("Range fetching failed, ignoring")
                  }
                }


                val divfactor = call.getOperands.get(1).asInstanceOf[RexLiteral].getValue3.asInstanceOf[java.math.BigDecimal];

                distinctRowCount = NumberUtil.min(overflow.divide(divfactor).doubleValue(), distinctRowCount)
              }
            } catch {
              case _: Throwable => System.err.println("Range fetching failed, ignoring")
            }
            // TODO zfong 6/21/06 - Broadbase has code to handle date
            // functions like year, month, day; E.g., cardinality of Month()
            // is 12
          }
          else {
            if (call.getOperands.size == 1) {
              distinctRowCount = cardOfProjExpr(mq, rel, call.getOperands.get(0))
            }
            else {
              distinctRowCount = rowCount / 10
            }
          }
        }
      }
      return numDistinctVals(distinctRowCount, rowCount)
    }
  }

  /**
   * Computes the cardinality of a particular expression from the projection
   * list.
   *
   * @param rel  RelNode corresponding to the project
   * @param expr projection expression
   * @return cardinality
   */
  def cardOfProjExpr(mq: RelMetadataQuery, rel: Project, expr: RexNode): java.lang.Double = expr.accept(new CardOfProjExpr(mq, rel))


  def getDistinctRowCount(rel: Project, mq: RelMetadataQuery, groupKey: ImmutableBitSet, predicate: RexNode): java.lang.Double = {
    if (predicate == null || predicate.isAlwaysTrue) if (groupKey.isEmpty) return 1D
    val baseCols = ImmutableBitSet.builder
    val projCols = ImmutableBitSet.builder
    val projExprs = rel.getProjects
    RelMdUtil.splitCols(projExprs, groupKey, baseCols, projCols)
    val notPushable = new util.ArrayList[RexNode]
    val pushable = new util.ArrayList[RexNode]
    RelOptUtil.splitFilters(ImmutableBitSet.range(rel.getRowType.getFieldCount), predicate, pushable, notPushable)
    val rexBuilder = rel.getCluster.getRexBuilder
    // get the distinct row count of the child input, passing in the
    // columns and filters that only reference the child; convert the
    // filter to reference the children projection expressions
    val childPred = RexUtil.composeConjunction(rexBuilder, pushable, true)
    var modifiedPred: RexNode = null
    if (childPred == null) modifiedPred = null
    else modifiedPred = RelOptUtil.pushPastProject(childPred, rel)
    var distinctRowCount = mq.getDistinctRowCount(rel.getInput, baseCols.build, modifiedPred)
    if (distinctRowCount == null) return null
    else if (!notPushable.isEmpty) {
      val preds = RexUtil.composeConjunction(rexBuilder, notPushable, true)
      distinctRowCount *= RelMdUtil.guessSelectivity(preds)
    }
    // No further computation required if the projection expressions
    // are all column references
    if (projCols.cardinality == 0) return distinctRowCount
    // multiply by the cardinality of the non-child projection expressions
    for (bit <- projCols.build.asScala) {
      val subRowCount = cardOfProjExpr(mq, rel, projExprs.get(bit))
      if (subRowCount == null) return null
      distinctRowCount *= subRowCount
    }
    RelMdUtil.numDistinctVals(distinctRowCount, mq.getRowCount(rel))
  }

  def getDistinctRowCount(rel: PelagoTableScan, mq: RelMetadataQuery,
                          groupKey: ImmutableBitSet, predicate: RexNode): java.lang.Double = {
    if (groupKey.isEmpty) return 1
    val cols = ImmutableBitSet.of(groupKey.asScala.map(e => rel.fields(e).asInstanceOf[java.lang.Integer]).asJava)
    val x = rel.getTable.unwrap(classOf[PelagoTable]).getDistrinctValues(cols)
    if (x == null) {
      RelMdUtil.numDistinctVals(
        cols.asScala
          .map(e => rel.getTable.unwrap(classOf[PelagoTable]).getDistrinctValues(ImmutableBitSet.of(e)))
          .reduce(_ * _), mq.getRowCount(rel)
      )
    } else {
      x
    }
  }
}
