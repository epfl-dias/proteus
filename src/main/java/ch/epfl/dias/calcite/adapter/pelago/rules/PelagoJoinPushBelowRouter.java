package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

public class PelagoJoinPushBelowRouter extends RelOptRule {
  public static final PelagoJoinPushBelowRouter INSTANCE = new PelagoJoinPushBelowRouter();

  protected PelagoJoinPushBelowRouter() {
    super(
      operand(
        PelagoJoin.class,
        operand(
          PelagoRouter.class,
          operand(RelNode.class, any())
        ),
        operand(
          PelagoRouter.class,
          operand(RelNode.class, any())
        )
      )
    );
  }

  public boolean matches(RelOptRuleCall call) {
    return true;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoJoin   join         =  (PelagoJoin  ) call.rel(0) ;
    PelagoRouter left_router  = ((PelagoRouter) call.rel(1));
    PelagoRouter right_router = ((PelagoRouter) call.rel(3));

    if (left_router.getDistribution() == RelDistributions.BROADCAST_DISTRIBUTED &&
        right_router.getDistribution() == RelDistributions.RANDOM_DISTRIBUTED) {
      RelNode lirouter = convert(call.rel(2), RelDeviceType.X86_64);
      RelNode rirouter = convert(call.rel(4), RelDeviceType.X86_64);

      PelagoJoin new_join = join.copy(null, join.getCondition(), lirouter, rirouter, join.getJoinType(), join.isSemiJoinDone());

      call.transformTo(PelagoRouter.create(new_join, RelDistributions.RANDOM_DISTRIBUTED));
    }
  }
}
