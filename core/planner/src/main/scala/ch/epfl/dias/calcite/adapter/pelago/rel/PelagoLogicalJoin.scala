package ch.epfl.dias.calcite.adapter.pelago.rel

import org.apache.calcite.plan.{Convention, RelOptCluster, RelTraitSet}
import org.apache.calcite.rel.core.{CorrelationId, Join, JoinRelType}
import org.apache.calcite.rel.{RelNode, RelWriter}
import org.apache.calcite.rex.RexNode

import java.util

class PelagoLogicalJoin(cluster: RelOptCluster, traitSet: RelTraitSet, left: RelNode, right: RelNode,
                        condition: RexNode, variablesSet: util.Set[CorrelationId], joinType: JoinRelType)
    extends Join(cluster, traitSet, left, right, condition, variablesSet, joinType) {
  override def copy(traitSet: RelTraitSet, conditionExpr: RexNode, left: RelNode, right: RelNode,
                    joinType: JoinRelType, semiJoinDone: Boolean): PelagoLogicalJoin = {
    new PelagoLogicalJoin(getCluster, getCluster.traitSetOf(Convention.NONE), left, right, conditionExpr, getVariablesSet, joinType)
  }
  
  override def explainTerms(pw: RelWriter): RelWriter = {
    val mq = getCluster.getMetadataQuery
    val leftRows = mq.getRowCount(getLeft)
    val rightRows = mq.getRowCount(getRight)
    val rows = Math.max(leftRows, rightRows)
    val small = if (leftRows < rightRows) getLeft else getRight
    val sel = mq.getPercentageOriginalRows(small)
    super.explainTerms(pw)
      .item("rct", mq.getRowCount(this) * sel)
      .item("sel", sel)
      .item("ht", left.getRowType.toString)
  }
}