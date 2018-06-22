package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.CorrelationId
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.metadata.{DefaultRelMetadataProvider, RelMdDistribution, RelMdParallelism, RelMetadataQuery}
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.{ImmutableIntList, Util}
import org.json4s.{JValue, JsonAST}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.Tuple2
import java.util

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
import com.google.common.base.Supplier

class PelagoJoin private (cluster: RelOptCluster, traitSet: RelTraitSet, left: RelNode, right: RelNode, condition: RexNode, variablesSet: util.Set[CorrelationId], joinType: JoinRelType)
//        assert getConvention() == left.getConvention();
//        assert getConvention() == right.getConvention();
//        assert !condition.isAlwaysTrue();
  extends Join(cluster, traitSet, left, right, condition, variablesSet, joinType) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, conditionExpr: RexNode, left: RelNode, right: RelNode, joinType: JoinRelType, semiJoinDone: Boolean) = {
    PelagoJoin.create(left, right, conditionExpr, getVariablesSet, joinType)
  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = super.estimateRowCount(mq)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = { // Pelago does not support cross products
    if (condition.isAlwaysTrue) return planner.getCostFactory.makeInfiniteCost

//    if (traitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX))) return planner.getCostFactory.makeTinyCost

//    if (getLeft.getRowType.getFieldCount > 1) return planner.getCostFactory.makeHugeCost
//    if (traitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX))) return planner.getCostFactory.makeTinyCost
//    var devFactor = if (traitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == RelDeviceType.NVPTX) 0.1 else 1

    var rowCount = mq.getRowCount(this)
    // Joins can be flipped, and for many algorithms, both versions are viable
    // and have the same cost. To make the results stable between versions of
    // the planner, make one of the versions slightly more expensive.
    //        switch (joinType) {
    //            case RIGHT:
    //                rowCount = addEpsilon(rowCount);
    //                break;
    //            default:
    //                if (RelNodes.COMPARATOR.compare(left, right) > 0) {
    //                    rowCount = addEpsilon(rowCount);
    //                }
    //        }
    // Cheaper if the smaller number of rows is coming from the LHS.
    // Model this by adding L log L to the cost.]
    val rightRowCount = right.estimateRowCount(mq)
    val leftRowCount = left.estimateRowCount(mq)

    if (leftRowCount.isInfinite) rowCount = leftRowCount
    else rowCount += Util.nLogN(leftRowCount * left.getRowType.getFieldCount);

    if (rightRowCount.isInfinite) {
      rowCount = rightRowCount
    } else {
      rowCount += rightRowCount //For the current HJ implementation, extra fields in the probing rel are 0-cost // * 0.1 * right.getRowType().getFieldCount();
      //TODO: Cost should change for radix-HJ
    }
    planner.getCostFactory.makeCost(rowCount, 0, 0).multiplyBy(0.1)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("build", left.getRowType.toString).item("lcount", Util.nLogN(left.estimateRowCount(left.getCluster.getMetadataQuery) * left.getRowType.getFieldCount)).item("rcount", right.estimateRowCount(right.getCluster.getMetadataQuery)).item("buildcountrow", left.estimateRowCount(left.getCluster.getMetadataQuery)).item("probecountrow", right.estimateRowCount(right.getCluster.getMetadataQuery))

//  override def estimateRowCount(mq: RelMetadataQuery): Double = mq.getRowCount(getRight) * mq.getPercentageOriginalRows(getLeft);//Math.max(mq.getRowCount(getLeft), mq.getRowCount(getRight))

  override def implement: (Binding, JsonAST.JValue) = {
    val op = ("operator" , "join")
    val l = getLeft.asInstanceOf[PelagoRel].implement()
    val leftBinding: Binding = l._1
    val leftChildOp = l._2
    val r = getRight.asInstanceOf[PelagoRel].implement()
    val rightBinding: Binding = r._1
    val rightChildOp = r._2
    val cond = emitExpression(getCondition, List(leftBinding,rightBinding))
    val alias = "join" + getId
    val rowType = emitSchema(alias, getRowType)

    val json = op ~ ("tupleType", rowType) ~ ("cond", cond) ~ ("left" , leftChildOp) ~ ("right" , rightChildOp)
    val binding: Binding = Binding(alias,leftBinding.fields ++ rightBinding.fields)
    val ret: (Binding, JValue) = (binding,json)
    ret
  }
}


object PelagoJoin {
  def create(left: RelNode, right: RelNode, condition: RexNode, variablesSet: util.Set[CorrelationId], joinType: JoinRelType) = {
    val cluster = right.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = right.getTraitSet.replace(PelagoRel.CONVENTION)
    new PelagoJoin(cluster, traitSet, left, right, condition, variablesSet, joinType)
  }
}