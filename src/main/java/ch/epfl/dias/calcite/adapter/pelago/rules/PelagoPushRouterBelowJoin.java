package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
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
//          any()
          operand(RelNode.class, any()),
          operand(RelNode.class, any())
        )
      )
    );
  }

  public boolean matches(RelOptRuleCall call) {
    return ((PelagoRouter) call.rel(0)).getDistribution() == RelDistributions.RANDOM_DISTRIBUTED;
  }

  public void onMatch(RelOptRuleCall call) {
//    PelagoRouter router = (PelagoRouter) call.rel(0);
    PelagoJoin   join   = call.rel(1);
//    RelNode      build  = join.getLeft ();//               call.rel(2);
//    RelNode      probe  = join.getRight();//               call.rel(3);

//    RelTraitSet btrait = build.getTraitSet().replace(RelDeviceType.X86_64).replace(RelDistributions.BROADCAST_DISTRIBUTED);
//    RelTraitSet ptrait = probe.getTraitSet().replace(RelDeviceType.X86_64).replace(RelDistributions.RANDOM_DISTRIBUTED);

//    RelNode new_build = convert(build, btrait);
//    RelNode new_probe = convert(probe, ptrait);

//    if (!build.getTraitSet().containsIfApplicable(RelDeviceType.X86_64)){
//      new_build = PelagoDeviceCross.create(new_build, build.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE));
//    }
//
//    if (!probe.getTraitSet().containsIfApplicable(RelDeviceType.X86_64)){
//      new_probe = PelagoDeviceCross.create(new_probe, probe.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE));
//    }

    RelNode      build  = call.rel(2);
    RelNode      probe  = call.rel(3);

    RelNode new_build = convert(build, RelDistributions.BROADCAST_DISTRIBUTED);

    RelNode new_probe = convert(probe, RelDistributions.RANDOM_DISTRIBUTED);


    call.transformTo(
      join.copy(
        null,
        join.getCondition(),
        new_build,
        new_probe,
        join.getJoinType(),
        join.isSemiJoinDone()
      )
    );
  }
}
