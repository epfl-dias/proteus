package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;

public class PelagoPushDeviceCrossNSplitBelowJoin extends RelOptRule {
  public static final PelagoPushDeviceCrossNSplitBelowJoin INSTANCE = new PelagoPushDeviceCrossNSplitBelowJoin();

  protected PelagoPushDeviceCrossNSplitBelowJoin() {
    super(
      operand(
        PelagoDeviceCross.class,
        operand(
          PelagoSplit.class,
          operand(
            PelagoJoin.class,
            operand(RelNode.class, any()),
            operand(RelNode.class, any())
          )
        )
      )
    );
  }

  public boolean matches(RelOptRuleCall call) {
    return true;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoDeviceCross dcross = (PelagoDeviceCross) call.rel(0);
    PelagoSplit       split  = (PelagoSplit      ) call.rel(1);
    PelagoJoin        join   = (PelagoJoin       ) call.rel(2);
    RelNode           build  =                     call.rel(3);
    RelNode           probe  =                     call.rel(4);

    call.transformTo(
      join.copy(
        null,
        join.getCondition(),
//        PelagoDeviceCross.create(build, dcross.getDeviceType()),
//        PelagoDeviceCross.create(probe, dcross.getDeviceType()),,
        convert(
          build,
          build.getTraitSet()
            .replace(RelHetDistribution.SPLIT_BRDCST)
            .replace(RelComputeDevice.from(dcross.getDeviceType()))
            .replace(dcross.getDeviceType())
        ),
        convert(
          probe,
          probe.getTraitSet()
            .replace(RelHetDistribution.SPLIT)
            .replace(RelComputeDevice.from(dcross.getDeviceType()))
            .replace(dcross.getDeviceType())
        ),
        join.getJoinType(),
        join.isSemiJoinDone()
      )
    );
  }
}
