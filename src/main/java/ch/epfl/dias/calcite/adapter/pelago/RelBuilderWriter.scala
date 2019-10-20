package ch.epfl.dias.calcite.adapter.pelago

import java.io.{PrintWriter, StringWriter}
import java.util

import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.rel.`type`.{RelDataType, RelDataTypeFactory}
import org.apache.calcite.rel.core.{Aggregate, AggregateCall, Filter, Join, Project, TableScan}
import org.apache.calcite.rel.externalize.RelWriterImpl
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.{RelNode, RelWriter}
import org.apache.calcite.rex.{RexCall, RexCorrelVariable, RexDynamicParam, RexFieldAccess, RexInputRef, RexLiteral, RexLocalRef, RexNode, RexOver, RexPatternFieldRef, RexRangeRef, RexShuttle, RexSubQuery, RexTableInputRef, RexUtil, RexVisitor, RexVisitorImpl}
import org.apache.calcite.sql.`type`.SqlTypeName
import org.apache.calcite.sql.fun.{SqlCountAggFunction, SqlSumAggFunction, SqlSumEmptyIsZeroAggFunction}
import org.apache.calcite.sql.{SqlExplainLevel, SqlKind}
import org.apache.calcite.util.Pair
import sun.jvm.hotspot.oops.ObjectHeap

import scala.collection.JavaConverters._

class RelBuilderWriter(pw: PrintWriter, detailLevel: SqlExplainLevel,
                       withIdPrefix: Boolean) extends RelWriterImpl(pw, detailLevel, withIdPrefix){
  def getPelagoRelBuilderName(rel: RelNode) = rel match {
    case _: TableScan => "scan"
    case _: Project => "project"
    case _: Filter => "filter"
    case _: Join => "join"
    case agg: Aggregate => if (agg.getGroupCount == 0) "reduce" else "groupby"
    case _: PelagoToEnumerableConverter => "print"
    case _: PelagoUnpack => "unpack"
    case _: PelagoPack => "pack"
    case router: PelagoRouter if router.getHomDistribution == RelHomDistribution.BRDCST => {
      "membrdcst(dop, true, true).router"
    }
    case _: PelagoRouter => "router"
  }

  protected def pre(rel: PelagoRel, s: java.lang.StringBuilder): Unit = rel match {
    case scan: PelagoTableScan =>
      s.append("<")
      .append("plugin_t") //TODO: scan.getPluginInfo.get("type"))
      .append(">")
    case _ =>
  }

  protected def canonicalizeTypes(e: RexNode, cluster: RelOptCluster): RexNode ={

    e.accept(new RexShuttle{
      override def visitCall(call: RexCall): RexNode = {
        val operands = super.visitCall(call).asInstanceOf[RexCall].getOperands
        val types = RexUtil.types(operands).asScala.map(cluster.getTypeFactory.createTypeWithNullability(_, false))
        if (!types.forall(_ == types.head)){
          val newType = cluster.getTypeFactory.leastRestrictive(types.asJava)
          call.clone(call.getType, operands.asScala.map(e => {
            if (cluster.getTypeFactory.createTypeWithNullability(e.getType, false).equals(newType)) {
              e
            } else {
              cluster.getRexBuilder.makeCast(cluster.getTypeFactory.createTypeWithNullability(newType, e.getType.isNullable), e)
            }
          }).asJava)
        } else {
          call
        }
      }
    })
  }

  protected def callexpr(e: RexCall, s: java.lang.StringBuilder, cluster: RelOptCluster): Unit = {
//    leastRestrictive
    var operands = e.getOperands.asScala.toList
    e.getKind match {
      case SqlKind.AND => {
        s.append("(")
        operands.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(" & ")
          expr(e._1, s, cluster)
        })
        s.append(")")
      }
      case SqlKind.OR => {
        s.append("(")
        operands.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(" | ")
          expr(e._1, s, cluster)
        })
        s.append(")")
      }
      case SqlKind.GREATER_THAN_OR_EQUAL => {
        s.append("ge(")
        append(operands, s, cluster)
        s.append(")")
      }
      case SqlKind.LESS_THAN_OR_EQUAL => {
        s.append("le(")
        append(operands, s, cluster)
        s.append(")")
      }
      case SqlKind.GREATER_THAN => {
        s.append("gt(")
        append(operands, s, cluster)
        s.append(")")
      }
      case SqlKind.LESS_THAN => {
        s.append("lt(")
        append(operands, s, cluster)
        s.append(")")
      }
      case SqlKind.EQUALS => {
        s.append("eq(")
        append(operands, s, cluster)
        s.append(")")
      }
      case SqlKind.NOT_EQUALS => {
        s.append("ne(")
        append(operands, s, cluster)
        s.append(")")
      }
      case SqlKind.CAST => {
        s.append("cast(")
        append(operands, s, cluster)
        s.append(")")
      }
    }
  }

  protected def expr(e: RexNode, s: java.lang.StringBuilder, cluster: RelOptCluster): Unit = e match {
    case call: RexCall => {
      callexpr(call, s, cluster)
    }
    case inp: RexInputRef => {
      s.append("arg[\"")
        .append(inp.getName)
        .append("\"]")
    }
    case lit: RexLiteral => {
      lit.getType.getSqlTypeName match {
        case SqlTypeName.DOUBLE | SqlTypeName.DECIMAL => {
          s.append("((double) ").append(lit.getValue()).append(")")
        }
        case SqlTypeName.INTEGER => s.append(lit)
      }
    }
  }

  protected def convAggInput(agg: AggregateCall, s: java.lang.StringBuilder) = agg.getAggregation match {
    case _: SqlSumAggFunction => {
      assert(agg.getArgList.size() == 1)
      s.append("arg[\"$")
        .append(agg.getArgList.get(0))
        .append("\"]")
    }
    case _: SqlCountAggFunction | _ : SqlSumEmptyIsZeroAggFunction => {
      assert(agg.getArgList.size() <= 1) //FIXME: nulls
      s.append("1") //TODO: does it require brackets?
    }
  }

  protected def convAggMonoid(agg: AggregateCall, s: java.lang.StringBuilder) = agg.getAggregation match {
    case _: SqlSumAggFunction => s.append("SUM")
    case _: SqlCountAggFunction | _ : SqlSumEmptyIsZeroAggFunction => s.append("SUM")
  }
  protected def append(x: Any, s: java.lang.StringBuilder, cluster: RelOptCluster): Unit = x match{
    case e: RexNode => expr(canonicalizeTypes(e, cluster), s, cluster)
    case l: util.List[_] => {
      l.asScala.zipWithIndex.foreach(e => {
        if (e._2 != 0) s.append(", ")
        append(e._1, s, cluster)
      })
    }
    case l: List[_] => {
      l.zipWithIndex.foreach(e => {
        if (e._2 != 0) s.append(", ")
        append(e._1, s, cluster)
      })
    }
    case str: String => {
      s.append("\"").append(str).append("\"")
    }
    case i: java.lang.Number => {
      s.append(i)
    }
  }

  protected def construction(rel: PelagoRel, s: java.lang.StringBuilder, values: Map[String, AnyRef]): Unit = {
    pre(rel, s)
    s.append("(")
    rel match {
      case scan: PelagoTableScan => {
        s.append("\"")
        s.append(scan.getPelagoRelName)
        s.append("\", {")
        scan.getRowType.getFieldList.asScala.map(field => {
          if (field.getIndex != 0) s.append(", ")
          s.append("\"" + field.getName + "\"")
        })
        s.append("}, catalog")
      }
      case _: PelagoUnpack =>
      case _: PelagoPack =>
      case filt: PelagoFilter => {
        s.append("[&](const auto &arg) -> expression_t { return ")
        append(filt.getCondition, s, rel.getCluster)
        s.append("; }")
      }
      case proj: PelagoProject => {
        s.append("[&](const auto &arg) -> std::vector<expression_t> { return {")
//        val names = proj.getRowType..asScala
        proj.getProjects.asScala.zipWithIndex.foreach(e => {
          if (e._2 != 0) s.append(", ")
          s.append("(")
          append(e._1, s, rel.getCluster)
          s.append(").as(")
          append(proj.getDigest, s, rel.getCluster)
          s.append(", \"$").append(e._2)
          s.append("\")")
        })
        s.append("}; }")
      }
      case join: PelagoJoin => {
        s.append("rel" + join.getLeft.getId)
        val buildKeys = join.getLeftKeys
        val probeKeys = join.getRightKeys
        s.append(", [&](const auto &build_arg) -> expression_t { return ")
        if (buildKeys.size() > 1) s.append("{")
        for (b <- buildKeys.asScala){
          s.append("build_arg[\"$").append(b).append("\"].as(\"")
          s.append(join.getDigest).append("\", \"")
          s.append("bk_").append(b.intValue()).append("\")")
        }
        if (buildKeys.size() > 1) s.append("}")
        s.append("; }, [&](const auto &probe_arg) -> expression_t { return ")
        if (probeKeys.size() > 1) s.append("{")
        for (p <- probeKeys.asScala){
          s.append("probe_arg[\"$").append(p).append("\"].as(\"")
          s.append(join.getDigest).append("\", \"")
          s.append("pk_").append(p.intValue()).append("\")")
        }
        if (probeKeys.size() > 1) s.append("}")
        s.append("; }, 10")
        s.append(", 1024 * 1024")
      }
      case agg: PelagoAggregate if (agg.getGroupCount == 0) => {
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
      }
      case router: PelagoRouter if router.getHomDistribution == RelHomDistribution.BRDCST => {
        val slack = 1
        s.append("[&](const auto &arg) -> std::optional<expression_t> { return arg[\"__broadcastTarget\"]; }, ")
        s.append("dop, ").append(slack).append(", RoutingPolicy::HASH_BASED, DeviceType::CPU, std::move(aff_parallel)")
      }
      case router: PelagoRouter if router.getHomDistribution == RelHomDistribution.RANDOM => {
        val slack = 1
        s.append("dop, ").append(slack).append(", RoutingPolicy::LOCAL, DeviceType::CPU, std::move(aff_parallel)")
      }
      case router: PelagoRouter if router.getHomDistribution == RelHomDistribution.SINGLE => {
        val slack = 128
        s.append("DegreeOfParallelism{1}, ").append(slack).append(", RoutingPolicy::RANDOM, DeviceType::CPU, std::move(aff_reduce)")
      }
      case _ => /* TODO: implement */
    }
    s.append(")")
  }

  protected def explainInputs(rel: RelNode){
    rel.getInputs().asScala.zipWithIndex.foreach(input => {
      val notlast = input._2 != rel.getInputs().size() - 1
      if (notlast) pw.print("auto rel" + input._1.getId + " = ")
      input._1.explain(this)
      if (notlast) pw.println(";")
    })
  }

  protected override def explain_(rel: RelNode, values: util.List[Pair[String, AnyRef]]): Unit = {
    val inputs = rel.getInputs
    val mq = rel.getCluster.getMetadataQuery
    if (!mq.isVisibleInExplain(rel, detailLevel)) { // render children in place of this, at same level
      explainInputs(rel)
      return
    }
    explainInputs(rel)
    if (inputs.isEmpty) {
      spacer.set(0)
      pw.println("RelBuilder{ctx}")
    }
    val s = new java.lang.StringBuilder
    spacer.spaces(s)
    if (withIdPrefix) s.append("/*id=").append(rel.getId).append("*/")
    s.append(".")
    s.append(getPelagoRelBuilderName(rel))
    rel match {
      case p: PelagoRel => construction(p, s, values.asScala.map(e => (e.left, e.right)).toMap)
      case root: PelagoToEnumerableConverter => {
        s.append("([&](const auto &arg, std::string outrel) -> std::vector<expression_t> { return {")
        //        val names = proj.getRowType..asScala
        root.getRowType.getFieldList.asScala.foreach(e => {
          if (e.getIndex != 0) s.append(", ")
          s.append("arg[\"$").append(e.getIndex).append("\"].as(")
          s.append("outrel")
          s.append(", ")
          append(e.getName, s, rel.getCluster)
          s.append(")")
        })
        s.append("}; })")
      }
      case _ => s.append("()")
    }
    s.append("\n // ")
    if (detailLevel ne SqlExplainLevel.NO_ATTRIBUTES) {
      var j = 0
      for (value <- values.asScala) {
        if (!value.right.isInstanceOf[RelNode]) {
          if ( {
            j += 1;
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
        s.append(": rowcount = ").append(mq.getRowCount(rel)).append(", cumulative cost = ").append(mq.getCumulativeCost(rel))
      case _ =>
    }
    detailLevel match {
      case SqlExplainLevel.NON_COST_ATTRIBUTES | SqlExplainLevel.ALL_ATTRIBUTES =>
        if (!withIdPrefix) { // If we didn't print the rel id at the start of the line, print
          // it at the end.
          s.append(", id = ").append(rel.getId)
        }
      case _ =>
    }
    pw.println(s)
    spacer.add(2)
//    spacer.subtract(2)
  }
}
