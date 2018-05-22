//package ch.epfl.dias.calcite.adapter.pelago;
//
//import ch.epfl.dias.emitter.PlanToJSON;
//import org.apache.calcite.plan.RelOptCluster;
//import org.apache.calcite.plan.RelOptCost;
//import org.apache.calcite.plan.RelOptPlanner;
//import org.apache.calcite.plan.RelTraitSet;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.core.Aggregate;
//import org.apache.calcite.rel.core.AggregateCall;
//import org.apache.calcite.rel.core.Project;
//import org.apache.calcite.rel.metadata.RelMetadataQuery;
//import org.apache.calcite.rel.type.RelDataType;
//import org.apache.calcite.rex.RexNode;
//import org.apache.calcite.util.ImmutableBitSet;
//import org.json4s.JsonAST;
//import scala.Tuple2;
//
//import java.util.List;
//
//public class PelagoAggregate extends Aggregate implements PelagoRel {
//    public PelagoAggregate(RelOptCluster cluster, RelTraitSet traitSet, RelNode input, boolean indicator,
//                           ImmutableBitSet groupSet, List<ImmutableBitSet> groupSets, List<AggregateCall> aggCalls) {
//        super(cluster, traitSet, input, indicator, groupSet, groupSets, aggCalls);
//        assert getConvention() == PelagoRel.CONVENTION;
//        assert getConvention() == input.getConvention();
//    }
//
//    @Override
//    public Aggregate copy(RelTraitSet traitSet, RelNode input, boolean indicator, ImmutableBitSet groupSet,
//                          List<ImmutableBitSet> groupSets, List<AggregateCall> aggCalls) {
//        return new PelagoAggregate(getCluster(), traitSet, input, indicator, groupSet, groupSets, aggCalls);
//    }
//
//    @Override public RelOptCost computeSelfCost(RelOptPlanner planner,
//                                                RelMetadataQuery mq) {
//        return super.computeSelfCost(planner, mq).multiplyBy(0.1);
//    }
//
//    @Override
//    public Tuple2<PlanToJSON.Binding, JsonAST.JValue> implement() {
//        return null;
//    }
//}
