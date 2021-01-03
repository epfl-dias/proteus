package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHomDistribution;

public class PelagoPushRouterBelowJoin extends PelagoPushRouterDown {
  public static final PelagoPushRouterBelowJoin INSTANCE = new PelagoPushRouterBelowJoin();

  protected PelagoPushRouterBelowJoin() {
    super(
      PelagoJoin.class,
      RelHomDistribution.RANDOM
    );
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoJoin   join   = call.rel(0);
    RelNode      build  = join.getLeft();
    RelNode      probe  = join.getRight();

    RelNode new_build = route(
      build,
      RelHomDistribution.BRDCST
    );

    RelNode new_probe = route(
      probe,
      trgt
    );

    call.transformTo(
      join.copy(
        null,
        ImmutableList.of(new_build, new_probe)
      )
    );
  }
}
