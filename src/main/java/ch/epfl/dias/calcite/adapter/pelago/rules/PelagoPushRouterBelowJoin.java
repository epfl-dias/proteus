package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableMap;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;

public class PelagoPushRouterBelowJoin extends RelOptRule {
  public static final PelagoPushRouterBelowJoin INSTANCE = new PelagoPushRouterBelowJoin();

  protected PelagoPushRouterBelowJoin() {
    super(
      operand(
        PelagoRouter.class,
        operand(
          PelagoJoin.class,
          any()
        )
      )
    );
  }

  public boolean matches(RelOptRuleCall call) {
    return ((PelagoRouter) call.rel(0)).getDistribution() == RelDistributions.RANDOM_DISTRIBUTED;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoJoin   join   = call.rel(1);
    RelNode      build  = join.getLeft();
    RelNode      probe  = join.getRight();

    RelNode new_build = PelagoRouter.create(
      convert(build, RelDeviceType.X86_64),
      RelDistributions.BROADCAST_DISTRIBUTED
    );

    RelNode new_probe = PelagoRouter.create(
      convert(probe, RelDeviceType.X86_64),
      RelDistributions.RANDOM_DISTRIBUTED
    );

    call.getPlanner().ensureRegistered(new_build, build);
    call.getPlanner().ensureRegistered(new_probe, probe);

    call.transformTo(
      join.copy(
        null,
        join.getCondition(),
        convert(new_build, join.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)),
        convert(new_probe, join.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)),
        join.getJoinType(),
        join.isSemiJoinDone()
      )
    );
  }
}
