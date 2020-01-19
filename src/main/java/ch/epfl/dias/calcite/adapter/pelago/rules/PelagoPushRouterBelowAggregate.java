package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRuleCall;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;

public class PelagoPushRouterBelowAggregate extends PelagoPushRouterDown {
  public static final PelagoPushRouterBelowAggregate INSTANCE = new PelagoPushRouterBelowAggregate();

  protected PelagoPushRouterBelowAggregate() {
    super(PelagoAggregate.class, RelHomDistribution.RANDOM);
  }

  public boolean matches(RelOptRuleCall call){
    if (!super.matches(call)) return false;

    PelagoAggregate rel = call.rel(0);
    return !rel.isGlobalAgg();
  }
}
