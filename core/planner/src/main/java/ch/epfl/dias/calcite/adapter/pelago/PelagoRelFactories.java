package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.Contexts;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.core.CorrelationId;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.hint.RelHint;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexCorrelVariable;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.rel.core.RelFactories.AggregateFactory;
import org.apache.calcite.rel.core.RelFactories.FilterFactory;
import org.apache.calcite.rel.core.RelFactories.JoinFactory;
import org.apache.calcite.rel.core.RelFactories.ProjectFactory;
import org.apache.calcite.rel.core.RelFactories.SemiJoinFactory;
import org.apache.calcite.rel.core.RelFactories.SetOpFactory;
import org.apache.calcite.rel.core.RelFactories.SortFactory;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class PelagoRelFactories {
  public static final ProjectFactory    PELAGO_PROJECT_FACTORY        = new PelagoProjectFactoryImpl();
  public static final FilterFactory     PELAGO_FILTER_FACTORY         = new PelagoFilterFactoryImpl();
  public static final JoinFactory       PELAGO_JOIN_FACTORY           = new PelagoJoinFactoryImpl();
//  public static final SemiJoinFactory   PELAGO_SEMI_JOIN_FACTORY      = new RelBuilderFactory.LogicalSemiJoinFactoryImpl();
  public static final SortFactory       PELAGO_SORT_FACTORY           = new PelagoSortFactoryImpl();
  public static final AggregateFactory  PELAGO_AGGREGATE_FACTORY      = new PelagoAggregateFactoryImpl();
//  public static final SetOpFactory      PELAGO_SET_OP_FACTORY         = new PelagoSetOpFactoryImpl();

  public static final RelBuilderFactory PELAGO_BUILDER                = PelagoRelBuilder.proto(
                                                                          Contexts.of(
                                                                              PELAGO_PROJECT_FACTORY  ,
                                                                              PELAGO_FILTER_FACTORY   ,
                                                                              PELAGO_JOIN_FACTORY     ,
////                                                                              PELAGO_SEMI_JOIN_FACTORY,
                                                                              PELAGO_SORT_FACTORY     ,
                                                                              PELAGO_AGGREGATE_FACTORY
//                                                                              PELAGO_SET_OP_FACTORY   ,
                                                                          )
                                                                        );


  private PelagoRelFactories() {
  }


  private static class PelagoProjectFactoryImpl implements ProjectFactory {

    @Override public RelNode createProject(final RelNode input, final List<RelHint> hints,
        final List<? extends RexNode> projects,
        final List<String> fieldNames) {
      RelOptCluster cluster = input.getCluster();
      RelDataType   rowType = RexUtil.createStructType(cluster.getTypeFactory(), projects, fieldNames);
      return PelagoProject.create(input, projects, rowType);
    }
  }

  private static class PelagoFilterFactoryImpl implements FilterFactory {
    @Override public RelNode createFilter(final RelNode input, final RexNode condition,
        final Set<CorrelationId> variablesSet) {
      return PelagoFilter.create(input, condition);
    }
  }

  private static class PelagoJoinFactoryImpl implements JoinFactory {
    @Override public RelNode createJoin(final RelNode left, final RelNode right, final List<RelHint> hints,
        final RexNode condition,
        final Set<CorrelationId> variablesSet, final JoinRelType joinType, final boolean semiJoinDone) {
      return PelagoJoin.create(left, right, condition, variablesSet, joinType);
    }
  }

  private static class PelagoSortFactoryImpl implements SortFactory {
    @Override
    public RelNode createSort(RelNode input, RelCollation collation, RexNode offset, RexNode fetch) {
      return PelagoSort.create(input, collation, offset, fetch);
    }
  }

  private static class PelagoAggregateFactoryImpl implements AggregateFactory {
    @Override public RelNode createAggregate(final RelNode input, final List<RelHint> hints,
        final ImmutableBitSet groupSet,
        final ImmutableList<ImmutableBitSet> groupSets, final List<AggregateCall> aggCalls) {
      return PelagoAggregate.create(input, ImmutableList.copyOf(hints), groupSet, groupSets, aggCalls);
    }
  }
}
