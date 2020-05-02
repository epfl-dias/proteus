package org.apache.calcite.plan;

import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class PelagoRelOptCluster extends RelOptCluster {
  PelagoRelOptCluster(final RelOptPlanner planner,
      final RelDataTypeFactory typeFactory,
      final RexBuilder rexBuilder, final AtomicInteger nextCorrel,
      final Map<String, RelNode> mapCorrelToRel) {
    super(planner, typeFactory, rexBuilder, nextCorrel, mapCorrelToRel);
  }

  PelagoRelMetadataQuery mq;

  /** Returns the current RelMetadataQuery.
   *
   * <p>This method might be changed or moved in future.
   * If you have a {@link RelOptRuleCall} available,
   * for example if you are in a {@link RelOptRule#onMatch(RelOptRuleCall)}
   * method, then use {@link RelOptRuleCall#getMetadataQuery()} instead. */
  public RelMetadataQuery getMetadataQuery() {
    if (mq == null) mq = PelagoRelMetadataQuery.instance();
    return mq;
  }

  /**
   * Should be called whenever the current {@link RelMetadataQuery} becomes
   * invalid. Typically invoked from {@link RelOptRuleCall#transformTo}.
   */
  public void invalidateMetadataQuery() {
    mq = null;
  }

  public static RelOptCluster create(RelOptPlanner planner,
      RexBuilder rexBuilder) {
    return new PelagoRelOptCluster(planner, rexBuilder.getTypeFactory(),
        rexBuilder, new AtomicInteger(0), new HashMap<String, RelNode>());
  }
}
