//package ch.epfl.dias.calcite.adapter.pelago.logical;
//
//import org.apache.calcite.plan.RelOptCluster;
//import org.apache.calcite.plan.RelTraitSet;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.core.AggregateCall;
//import org.apache.calcite.rel.logical.LogicalAggregate;
//import org.apache.calcite.util.ImmutableBitSet;
//
//import java.util.List;
//
//public class PelagoLogicalAggregate extends LogicalAggregate {
//  public PelagoLogicalAggregate(RelOptCluster cluster, RelTraitSet traitSet, RelNode child, boolean indicator, ImmutableBitSet groupSet, List<ImmutableBitSet> groupSets, List<AggregateCall> aggCalls) {
//    super(cluster, traitSet, child, indicator, groupSet, groupSets, aggCalls);
//  }
//
//}
