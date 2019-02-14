package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;

public class PelagoPushDeviceCrossBelowJoin extends RelOptRule {
  public static final PelagoPushDeviceCrossBelowJoin INSTANCE = new PelagoPushDeviceCrossBelowJoin();

  protected PelagoPushDeviceCrossBelowJoin() {
    super(
      operand(
        PelagoDeviceCross.class,
        operand(
          PelagoJoin.class,
          any()
        )
      )
    );
  }

  public boolean matches(RelOptRuleCall call) {
    return true;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoDeviceCross dcross = (PelagoDeviceCross) call.rel(0);
    PelagoJoin        join   = (PelagoJoin       ) call.rel(1);
    RelNode           build  =                     join.getLeft();
    RelNode           probe  =                     join.getRight();

    call.transformTo(
      join.copy(
        null,
        join.getCondition(),
//        PelagoDeviceCross.create(build, dcross.getDeviceType()),
//        PelagoDeviceCross.create(probe, dcross.getDeviceType()),
        convert(build, dcross.getDeviceType()),
        convert(probe, dcross.getDeviceType()),
        join.getJoinType(),
        join.isSemiJoinDone()
      )
    );
  }
}
