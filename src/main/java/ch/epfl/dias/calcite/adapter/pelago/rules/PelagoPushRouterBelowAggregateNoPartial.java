package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;

public class PelagoPushRouterBelowAggregateNoPartial extends PelagoPushRouterBelowAggregate {
  public static final PelagoPushRouterBelowAggregateNoPartial INSTANCE = new PelagoPushRouterBelowAggregateNoPartial();

  protected PelagoPushRouterBelowAggregateNoPartial() {
    super();
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoAggregate rel    = call.rel(0);

    if (rel.getGroupCount() == 0) {
      super.onMatch(call);
      return;
    }

    RelNode locagg = convert(
      convert(rel.getInput(), rel.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)),
      RelHomDistribution.RANDOM
    );

    RelNode agg = rel.copy(null, ImmutableList.of(
        convert(
            locagg,
            locagg.getTraitSet()
                .replace(RelHomDistribution.SINGLE)
                .replace(rel.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE))
        )
    ));

    call.getPlanner().ensureRegistered(agg, rel);

    call.transformTo(
        convert(agg, rel.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE))
    );
  }
}
