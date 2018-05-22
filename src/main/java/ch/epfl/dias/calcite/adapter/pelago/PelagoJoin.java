//package ch.epfl.dias.calcite.adapter.pelago;
//
//import ch.epfl.dias.emitter.Binding;
//import ch.epfl.dias.emitter.PlanToJSON;
//import org.apache.calcite.plan.RelOptCluster;
//import org.apache.calcite.plan.RelOptCost;
//import org.apache.calcite.plan.RelOptPlanner;
//import org.apache.calcite.plan.RelTraitSet;
//import org.apache.calcite.rel.RelDistributionTraitDef;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.RelNodes;
//import org.apache.calcite.rel.RelWriter;
//import org.apache.calcite.rel.core.CorrelationId;
//import org.apache.calcite.rel.core.Join;
//import org.apache.calcite.rel.core.JoinRelType;
//import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider;
//import org.apache.calcite.rel.metadata.RelMdParallelism;
//import org.apache.calcite.rel.metadata.RelMetadataQuery;
//import org.apache.calcite.rex.RexNode;
//import org.apache.calcite.util.Util;
//import org.json4s.JsonAST;
//import scala.Tuple2;
//
//import java.util.Set;
//
//public class PelagoJoin extends Join implements PelagoRel {
//    public PelagoJoin(
//            RelOptCluster cluster,
//            RelTraitSet traitSet,
//            RelNode left,
//            RelNode right,
//            RexNode condition,
//            Set<CorrelationId> variablesSet,
//            JoinRelType joinType) {
//        super(cluster, traitSet, left, right, condition, variablesSet, joinType);
//        assert getConvention() == PelagoRel.CONVENTION;
////        assert getConvention() == left.getConvention();
////        assert getConvention() == right.getConvention();
////        assert !condition.isAlwaysTrue();
//    }
//
//    @Override
//    public Join copy(RelTraitSet traitSet, RexNode conditionExpr, RelNode left, RelNode right, JoinRelType joinType, boolean semiJoinDone) {
//        return new PelagoJoin(getCluster(), traitSet, left, right, conditionExpr, getVariablesSet(), joinType);
//    }
//
//    @Override
//    public RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
//        // Pelago does not support cross products
//        if (condition.isAlwaysTrue()) return planner.getCostFactory().makeInfiniteCost();
//
//        double rowCount = mq.getRowCount(this);
//
//        // Joins can be flipped, and for many algorithms, both versions are viable
//        // and have the same cost. To make the results stable between versions of
//        // the planner, make one of the versions slightly more expensive.
////        switch (joinType) {
////            case RIGHT:
////                rowCount = addEpsilon(rowCount);
////                break;
////            default:
////                if (RelNodes.COMPARATOR.compare(left, right) > 0) {
////                    rowCount = addEpsilon(rowCount);
////                }
////        }
//
//        // Cheaper if the smaller number of rows is coming from the LHS.
//        // Model this by adding L log L to the cost.]
//
//        final double rightRowCount = right.estimateRowCount(mq);
//        final double leftRowCount = left.estimateRowCount(mq);
//        if (Double.isInfinite(leftRowCount)) {
//            rowCount = leftRowCount;
//        } else {
//            rowCount += Util.nLogN(leftRowCount * left.getRowType().getFieldCount());
//        }
//        if (Double.isInfinite(rightRowCount)) {
//            rowCount = rightRowCount;
//        } else {
//            rowCount += rightRowCount; //For the current HJ implementation, extra fields in the probing rel are 0-cost // * 0.1 * right.getRowType().getFieldCount();
//            //TODO: Cost should change for radix-HJ
//        }
//        return planner.getCostFactory().makeCost(rowCount, 0, 0).multiplyBy(0.1);
//    }
//
//    @Override
//    public RelWriter explainTerms(RelWriter pw) {
//        return super.explainTerms(pw).item("build", left.getRowType().toString())
//                .item("lcount", Util.nLogN(left.estimateRowCount(left.getCluster().getMetadataQuery()) * left.getRowType().getFieldCount()))
//                .item("rcount", right.estimateRowCount(right.getCluster().getMetadataQuery()))
//                .item("lcount2", Util.nLogN(right.estimateRowCount(right.getCluster().getMetadataQuery()) * right.getRowType().getFieldCount()))
//                .item("rcount2", left.estimateRowCount(left.getCluster().getMetadataQuery()));
//    }
//
//    @Override
//    public double estimateRowCount(RelMetadataQuery mq) {
//        return Math.max(mq.getRowCount(getLeft()), mq.getRowCount(getRight()));
//    }
//
//    @Override
//    public Tuple2<Binding, JsonAST.JValue> implement() {
//        return null;
//    }
//}
