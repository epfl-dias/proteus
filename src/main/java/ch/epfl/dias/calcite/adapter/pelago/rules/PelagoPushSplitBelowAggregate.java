package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;

public class PelagoPushSplitBelowAggregate extends PelagoPushSplitDown {
  public static final RelOptRule INSTANCE = new PelagoPushSplitBelowAggregate();

  protected PelagoPushSplitBelowAggregate() {
    super(PelagoAggregate.class, RelHetDistribution.SPLIT);
  }

  public boolean matches(RelOptRuleCall call){
    if (!super.matches(call)) return false;

    PelagoAggregate rel = call.rel(0);
    return !rel.isGlobalAgg();
  }
}
