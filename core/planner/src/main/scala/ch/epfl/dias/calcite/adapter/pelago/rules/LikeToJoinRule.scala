package ch.epfl.dias.calcite.adapter.pelago.rules

import ch.epfl.dias.calcite.adapter.pelago.rel.LogicalPelagoDictTableScan
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptRule.{any, operand}
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.{Filter, JoinRelType}
import org.apache.calcite.rex._
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.fun.SqlStdOperatorTable
import org.apache.calcite.tools.RelBuilder
import org.apache.calcite.util.NlsString

import scala.collection.JavaConverters._

object LikeToJoinRule { val INSTANCE = new LikeToJoinRule }

class LikeToJoinRule protected ()
    extends RelOptRule(operand(classOf[Filter], any)) {

  private[rules] class FindLikes(
      val builder: RexBuilder,
      var input: RelNode,
      var relBuilder: RelBuilder
  ) extends RexShuttle {

    private[rules] var cnt = input.getRowType.getFieldCount

    override def visitCall(call: RexCall): RexNode = {
      if (call.getKind eq SqlKind.LIKE) {
        val ref = input.getCluster.getMetadataQuery
          .getExpressionLineage(input, call.getOperands.get(0))
        assert(
          ref != null,
          "Have you forgot to add an operator in the expression lineage metadata provider?"
        )
        assert(ref.size == 1)

        val attrIndex =
          ref.iterator.next.asInstanceOf[RexTableInputRef].getIndex
        val regex = {
          call.getOperands.get(1).asInstanceOf[RexLiteral]
        }.getValue.asInstanceOf[NlsString].getValue

        val table =
          ref.iterator.next.asInstanceOf[RexTableInputRef].getTableRef.getTable

        table.getRelOptSchema.getTableForMember(table.getQualifiedName)
        input = relBuilder
          .push(input)
          .push(
            LogicalPelagoDictTableScan
              .create(input.getCluster, table, regex, attrIndex)
          )
          .join(JoinRelType.INNER, builder.makeLiteral(true))
          .build

        return builder.makeCall(
          SqlStdOperatorTable.EQUALS,
          call.getOperands.get(0),
          builder.makeInputRef(
            call.getOperands.get(0).getType,
            { cnt += 1; cnt - 1 }
          )
        )
      }
      super.visitCall(call)
    }
    def getNewInput: RelNode = input
  }

  override def onMatch(call: RelOptRuleCall): Unit = {
    val filter: Filter = call.rel(0)
    val cond = filter.getCondition
    val rexBuilder = filter.getCluster.getRexBuilder
    val input: RelNode = filter.getInput
    val fl = new FindLikes(rexBuilder, input, call.builder())
    val new_cond = cond.accept(fl)
    if (fl.getNewInput ne input) {
      // do not consider the matched node again!
      call.getPlanner.prune(filter)
      val projs = ImmutableList.builder[RexNode]

      for (f <- filter.getRowType.getFieldList.asScala) {
        projs.add(rexBuilder.makeInputRef(f.getType, f.getIndex))
      }
      val replacement = call.builder
        .push(fl.getNewInput)
        .filter(new_cond)
        .project(projs.build)
        .build

      // push transformation
      call.transformTo(replacement)
    }
  }
}
