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

package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.calcite.adapter.pelago.rel._
import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelDeviceType,
  RelHomDistribution
}
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core._
import org.apache.calcite.rel.externalize.RelWriterImpl
import org.apache.calcite.rex._
import org.apache.calcite.sql.`type`.SqlTypeName
import org.apache.calcite.sql.fun.{
  SqlCountAggFunction,
  SqlMinMaxAggFunction,
  SqlSumAggFunction,
  SqlSumEmptyIsZeroAggFunction
}
import org.apache.calcite.sql.{SqlExplainLevel, SqlKind}
import org.apache.calcite.util.Pair

import java.io.{PrintWriter, StringWriter}
import java.{lang, util}
import scala.collection.JavaConverters._

class RelBuilderWriter(
    pw: PrintWriter,
    detailLevel: SqlExplainLevel,
    withIdPrefix: Boolean
) extends RelWriterImpl(pw, detailLevel, withIdPrefix) {
  def getPelagoRelBuilderName(rel: RelNode): String =
    rel match {
      case _: TableScan                   => "template scan"
      case _: Project                     => "project"
      case _: Filter                      => "filter"
      case _: Join                        => "join"
      case agg: Aggregate                 => if (agg.getGroupCount == 0) "reduce" else "groupby"
      case _: PelagoToEnumerableConverter => "print"
      case _: PelagoUnpack                => "unpack"
      case _: PelagoPack                  => "pack"
      case router: PelagoRouter
          if router.getHomDistribution == RelHomDistribution.BRDCST =>
        "membrdcst(dop, true, true).router"
      case _: PelagoRouter => "router"
      case _: PelagoSort   => "sort"
      case dcross: PelagoDeviceCross =>
        if (dcross.getDeviceType() == RelDeviceType.X86_64) "to_cpu"
        else "to_gpu"
    }

  protected def pre(rel: PelagoRel, s: java.lang.StringBuilder): Unit =
    rel match {
      case _: PelagoTableScan =>
        s.append("<")
          .append("Tplugin") //TODO: scan.getPluginInfo.get("type"))
          .append(">")
      case _ =>
    }

  protected def canonicalizeTypes(
      e: RexNode,
      cluster: RelOptCluster
  ): RexNode = {

    e.accept(new RexShuttle {
      override def visitCall(call: RexCall): RexNode = {
        val operands = super.visitCall(call).asInstanceOf[RexCall].getOperands
        if (call.isA(SqlKind.EXTRACT)) return call
        val types = RexUtil
          .types(operands)
          .asScala
          .map(cluster.getTypeFactory.createTypeWithNullability(_, false))
        if (!types.forall(_ == types.head)) {
          val newType = cluster.getTypeFactory.leastRestrictive(types.asJava)
          call.clone(
            call.getType,
            operands.asScala
              .map(e => {
                if (
                  newType == null /* for example for CASE */ || cluster.getTypeFactory
                    .createTypeWithNullability(e.getType, false)
                    .equals(newType)
                ) {
                  e
                } else {
                  cluster.getRexBuilder.makeCast(
                    cluster.getTypeFactory
                      .createTypeWithNullability(newType, e.getType.isNullable),
                    e
                  )
                }
              })
              .asJava
          )
        } else {
          call
        }
      }
    })
  }

  protected def callexpr(
      e: RexCall,
      s: java.lang.StringBuilder,
      cluster: RelOptCluster
  ): Unit = {
    //    leastRestrictive
    val operands = e.getOperands.asScala.toList
    e.getKind match {
      case SqlKind.AND =>
        s.append("(")
        operands.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(" & ")
          expr(e._1, s, cluster)
        })
        s.append(")")
      case SqlKind.OR =>
        s.append("(")
        operands.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(" | ")
          expr(e._1, s, cluster)
        })
        s.append(")")
      case SqlKind.GREATER_THAN_OR_EQUAL =>
        s.append("ge(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.LESS_THAN_OR_EQUAL =>
        s.append("le(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.GREATER_THAN =>
        s.append("gt(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.LESS_THAN =>
        s.append("lt(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.EQUALS =>
        s.append("eq(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.NOT_EQUALS =>
        s.append("ne(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.CAST =>
        s.append("cast(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.CASE =>
        s.append("cond(")
        append(operands, s, cluster)
        s.append(")")
      case SqlKind.DIVIDE =>
        s.append("(")
        append(operands.head, s, cluster)
        s.append("/")
        append(operands(1), s, cluster)
        s.append(")")
      case SqlKind.MOD =>
        s.append("(")
        append(operands.head, s, cluster)
        s.append("%")
        append(operands(1), s, cluster)
        s.append(")")
      case SqlKind.MINUS =>
        s.append("(")
        append(operands.head, s, cluster)
        s.append("-")
        append(operands(1), s, cluster)
        s.append(")")
      case SqlKind.MINUS_PREFIX =>
        s.append("(-")
        append(operands.head, s, cluster)
        s.append(")")
      case SqlKind.TIMES =>
        s.append("(")
        append(operands.head, s, cluster)
        s.append("*")
        append(operands(1), s, cluster)
        s.append(")")
      case SqlKind.EXTRACT =>
        s.append("expressions::ExtractExpression{")
        append(operands(1), s, cluster)
        s.append(", expressions::extract_unit::").append(operands.head)
        s.append("}")
      case SqlKind.OTHER_FUNCTION =>
//        e.getOperator match {
//          case function: SqlFunction => function.getSqlIdentifier match {
//            case _ => ???
//          }
//        }
        ???
      case _ => ???
    }
  }

  protected def expr(
      e: RexNode,
      s: java.lang.StringBuilder,
      cluster: RelOptCluster
  ): Unit =
    e match {
      case call: RexCall =>
        callexpr(call, s, cluster)
      case inp: RexInputRef =>
        s.append("arg[\"")
          .append(inp.getName)
          .append("\"]")
      case lit: RexLiteral =>
        lit.getType.getSqlTypeName match {
          case SqlTypeName.DOUBLE | SqlTypeName.DECIMAL =>
            s.append("((double) ").append(lit.getValue).append(")")

          case SqlTypeName.INTEGER => s.append(lit)
          case SqlTypeName.BIGINT =>
            s.append("((int64_t) ").append(lit).append(")")
          case SqlTypeName.TIMESTAMP =>
            s.append("expressions::DateConstant(")
              .append({
                val sw = new StringWriter
                val pw = new PrintWriter(sw)
                pw.print('"')
                lit.printAsJava(pw)
                pw.print('"')
                pw.flush()
                sw.toString
              })
              .append(")")
          case SqlTypeName.VARCHAR | SqlTypeName.CHAR =>
            append(lit.getValue2, s, cluster)
          case SqlTypeName.SYMBOL =>
            s.append(lit.getValue)

          case SqlTypeName.DATE => ???
        }
    }

  protected def convAggInput(
      agg: AggregateCall,
      s: java.lang.StringBuilder
  ): lang.StringBuilder =
    agg.getAggregation match {
      case _: SqlSumAggFunction | _: SqlSumEmptyIsZeroAggFunction |
          _: SqlMinMaxAggFunction =>
        assert(agg.getArgList.size() == 1)
        s.append("arg[\"$")
          .append(agg.getArgList.get(0))
          .append("\"]")
      case _: SqlCountAggFunction =>
        assert(agg.getArgList.size() <= 1) //FIXME: nulls
        s.append("expression_t{1}")
    }

  protected def convAggMonoid(
      agg: AggregateCall,
      s: java.lang.StringBuilder
  ): lang.StringBuilder =
    agg.getAggregation match {
      case _: SqlSumAggFunction => s.append("SUM")
      case _: SqlCountAggFunction | _: SqlSumEmptyIsZeroAggFunction =>
        s.append("SUM")
      case a: SqlMinMaxAggFunction =>
        s.append(a.getKind match {
          case SqlKind.MIN => "MIN"
          case SqlKind.MAX => "MAX"
        })
    }
  protected def append(
      x: Any,
      s: java.lang.StringBuilder,
      cluster: RelOptCluster
  ): Unit =
    x match {
      case e: RexNode => expr(canonicalizeTypes(e, cluster), s, cluster)
      case l: util.List[_] =>
        l.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          append(e._1, s, cluster)
        })
      case l: List[_] =>
        l.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          append(e._1, s, cluster)
        })
      case str: String =>
        s.append("\"").append(str).append("\"")
      case i: java.lang.Number =>
        s.append(i)
    }

  protected def construction(
      rel: PelagoRel,
      s: java.lang.StringBuilder
  ): Unit = {
    pre(rel, s)
    s.append("(")
    rel match {
      case scan: PelagoTableScan =>
        s.append("\"")
        s.append(scan.getPelagoRelName)
        s.append("\", {")
        scan.getRowType.getFieldList.asScala.map(field => {
          if (field.getIndex != 0) s.append(", ")
          s.append("\"" + field.getName + "\"")
        })
        s.append("}, getCatalog()")
      case _: PelagoUnpack =>
      case _: PelagoPack   =>
      case filt: PelagoFilter =>
        s.append("[&](const auto &arg) -> expression_t { return ")
        append(filt.getCondition, s, rel.getCluster)
        s.append("; }")
      case proj: PelagoProject =>
        s.append("[&](const auto &arg) -> std::vector<expression_t> { return {")
        //        val names = proj.getRowType..asScala
        proj.getNamedProjects.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          s.append("(")
          append(e._1.left, s, rel.getCluster)
          s.append(").as(")
          append(proj.getDigest, s, rel.getCluster)
          s.append(", \"").append(e._1.right)
          s.append("\")")
        })
        s.append("}; }")
      case join: PelagoJoin =>
        s.append("rel" + join.getLeft.getId)
        val joinInfo = join.analyzeCondition()
        val buildKeys = joinInfo.leftKeys
        val probeKeys = joinInfo.rightKeys
        s.append(", [&](const auto &build_arg) -> expression_t { return ")
        if (buildKeys.size() > 1) s.append("expressions::RecordConstruction{")
        buildKeys.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          s.append("build_arg[\"$").append(e._1).append("\"].as(\"")
          s.append(join.getDigest).append("\", \"")
          s.append("bk_").append(e._1).append("\")")
        })
        if (buildKeys.size() > 1) {
          s.append("}").append(".as(\"")
          s.append(join.getDigest).append("\", \"bk\")")
        }
        s.append("; }, [&](const auto &probe_arg) -> expression_t { return ")
        if (probeKeys.size() > 1) s.append("expressions::RecordConstruction{")
        probeKeys.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          s.append("probe_arg[\"$").append(e._1).append("\"].as(\"")
          s.append(join.getDigest).append("\", \"")
          s.append("pk_").append(e._1).append("\")")
        })
        if (probeKeys.size() > 1) {
          s.append("}").append(".as(\"")
          s.append(join.getDigest).append("\", \"pk\")")
        }
        val rowEst = join.getCluster.getMetadataQuery.getRowCount(join.getLeft)
        val maxrow =
          join.getCluster.getMetadataQuery.getMaxRowCount(join.getLeft)
        val maxEst =
          if (maxrow != null) Math.min(maxrow, 64 * 1024 * 1024)
          else 64 * 1024 * 1024

        val hash_bits = Math.min(
          2 + Math.ceil(Math.log(rowEst) / Math.log(2)).asInstanceOf[Int],
          28
        )

        s.append("; }, ").append(hash_bits.asInstanceOf[Int])
        s.append(", ").append(maxEst.asInstanceOf[Long])
      case agg: PelagoAggregate if agg.getGroupCount == 0 =>
        s.append("[&](const auto &arg) -> std::vector<expression_t> { return {")
        //        val names = proj.getRowType..asScala
        agg.getAggCallList.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          s.append("(")
          convAggInput(e._1, s)
          s.append(").as(")
          append(agg.getDigest, s, rel.getCluster)
          s.append(", \"$").append(e._2)
          s.append("\")")
        })
        s.append("}; }, {")
        //        val names = proj.getRowType..asScala
        agg.getAggCallList.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          convAggMonoid(e._1, s)
        })
        s.append("}")
      case agg: PelagoAggregate if agg.getGroupCount > 0 =>
        s.append("[&](const auto &arg) -> std::vector<expression_t> { return {")
        //        val names = proj.getRowType..asScala
        agg.getGroupSet.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          s.append("arg[\"$")
          s.append(e._1)
          s.append("\"].as(")
          append(agg.getDigest, s, rel.getCluster)
          s.append(", \"$").append(e._2)
          s.append("\")")
        })
        s.append("}; }, ")
        s.append(
          "[&](const auto &arg) -> std::vector<GpuAggrMatExpr> { return {"
        )
        //        val names = proj.getRowType..asScala
        agg.getAggCallList.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          s.append("GpuAggrMatExpr{(")
          convAggInput(e._1, s)
          s.append(").as(")
          append(agg.getDigest, s, rel.getCluster)
          s.append(", \"$").append(e._2 + agg.getGroupCount)
          s.append("\"), ").append(e._2 + 1)
          s.append(", 0, ")
          convAggMonoid(e._1, s)
          s.append("}")
        })
        //1 vs 128 vs 64
        val maxEst =
          131072 //if (maxrow != null) Math.min(maxrow, 32*1024*1024) else 32*1024*1024 //1 vs 128 vs 64

        val hash_bits =
          10 //Math.min(1 + Math.ceil(Math.log(rowEst)/Math.log(2)).asInstanceOf[Int], 10)

        s.append("}; }, ").append(hash_bits).append(", ").append(maxEst)
      case router: PelagoRouter
          if router.getHomDistribution == RelHomDistribution.BRDCST =>
        val slack = 64
        s.append(
          "[&](const auto &arg) -> std::optional<expression_t> { return arg[\"__broadcastTarget\"]; }, "
        )
        s.append("dop, ")
          .append(slack)
          .append(", RoutingPolicy::HASH_BASED, dev, aff_parallel()")
      case router: PelagoRouter
          if router.getHomDistribution == RelHomDistribution.RANDOM =>
        val slack = 32
        s.append("dop, ")
          .append(slack)
          .append(", RoutingPolicy::LOCAL, dev, aff_parallel()")
      case router: PelagoRouter
          if router.getHomDistribution == RelHomDistribution.SINGLE =>
        val slack = 128
        s.append("DegreeOfParallelism{1}, ")
          .append(slack)
          .append(", RoutingPolicy::RANDOM, DeviceType::CPU, aff_reduce()")
      case sort: PelagoSort =>
        s.append(
          "[&](const auto &arg) -> std::vector<expression_t> { return {"
        )
        (0 until sort.getRowType.getFieldCount).foreach(e => {
          if (e != 0) s.append(", ")
          s.append("arg[\"$").append(e).append("\"]")
        })
        s.append("}; }, {")
        sort.getCollation.getFieldCollations.asScala.zipWithIndex.foreach(e => {
          s.append("direction::")
            .append(e._1.direction.shortString)
            .append(", ")
        })
        (0 until (sort.getRowType.getFieldCount - sort.getCollation.getFieldCollations.size))
          .foreach(_ => {
            s.append("direction::NONE, ")
          })
        s.append("}")
      case _: PelagoDeviceCross =>
      case _: PelagoDictTableScan =>
        s.append(". /* Unimplemeted node */");
      //      case _ => /* TODO: implement */
    }
    s.append(")")
  }

  protected def explainInputs(rel: RelNode) {
    rel.getInputs.asScala.zipWithIndex
      .foreach(input => {
        val notlast = input._2 != rel.getInputs.size() - 1
        if (notlast) pw.print("auto rel" + input._1.getId + " = ")
//      else if (rel.isInstanceOf[PelagoToEnumerableConverter]) pw.print("return ")
        input._1.explain(this)
        if (notlast) pw.println(";")
      })
  }

  protected override def explain_(
      rel: RelNode,
      values: util.List[Pair[String, AnyRef]]
  ): Unit = {
    val inputs = rel.getInputs
    val mq = rel.getCluster.getMetadataQuery
    if (!mq.isVisibleInExplain(rel, detailLevel)) { // render children in place of this, at same level
      explainInputs(rel)
      return
    }
    explainInputs(rel)
    if (inputs.isEmpty) {
      spacer.set(0)
      pw.println("getBuilder<Tplugin>()")
    }
    val s = new java.lang.StringBuilder
    spacer.spaces(s)
    if (withIdPrefix) s.append("/*id=").append(rel.getId).append("*/")
    s.append(".")
    s.append(getPelagoRelBuilderName(rel))
    rel match {
      case p: PelagoRel =>
        construction(p, s)
      case _: PelagoToEnumerableConverter =>
        s.append("(pg{\"pm-csv\"})")
      case _ => s.append("()")
    }
    s.append(" // ")
    if (detailLevel ne SqlExplainLevel.NO_ATTRIBUTES) {
      var j = 0
      for (value <- values.asScala) {
        if (!value.right.isInstanceOf[RelNode]) {
          if ({
            j += 1
            j - 1
          } == 0) s.append("(")
          else s.append(", ")
          s.append(value.left).append("=[").append(value.right).append("]")
        }
      }
      if (j > 0) s.append(")")
    }
    detailLevel match {
      case SqlExplainLevel.ALL_ATTRIBUTES =>
        s.append(": rowcount = ")
          .append(mq.getRowCount(rel))
          .append(", cumulative cost = ")
          .append(mq.getCumulativeCost(rel))
      case _ =>
    }
    detailLevel match {
      case SqlExplainLevel.NON_COST_ATTRIBUTES |
          SqlExplainLevel.ALL_ATTRIBUTES =>
        if (!withIdPrefix) { // If we didn't print the rel id at the start of the line, print
          // it at the end.
          s.append(", id = ").append(rel.getId)
        }
      case _ =>
    }
    if (rel.isInstanceOf[PelagoToEnumerableConverter]) {
      s.append("\n")
      spacer.spaces(s)
      s.append(".prepare();")
    }
    pw.println(s)
    spacer.add(2)
//    spacer.subtract(2)
  }
}
