package org.apache.calcite.plan;

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataProvider;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class PelagoRelOptCluster extends RelOptCluster {
  PelagoRelOptCluster(final RelOptPlanner planner,
      final RelDataTypeFactory typeFactory,
      final RexBuilder rexBuilder, final AtomicInteger nextCorrel,
      final Map<String, RelNode> mapCorrelToRel) {
    super(planner, typeFactory, rexBuilder, nextCorrel, mapCorrelToRel);
    super.setMetadataProvider(PelagoRelMetadataProvider.INSTANCE);
    setMetadataQuerySupplier(() -> {
      super.setMetadataProvider(ChainedRelMetadataProvider.of(List.of(getMetadataProvider(),
          PelagoRelMetadataProvider.INSTANCE)));
      return PelagoRelMetadataQuery.instance();
    });
  }

  public static RelOptCluster create(RelOptPlanner planner,
                                     RexBuilder rexBuilder) {
    return new PelagoRelOptCluster(planner, rexBuilder.getTypeFactory(),
        rexBuilder, new AtomicInteger(0), new HashMap<String, RelNode>());
  }
}
