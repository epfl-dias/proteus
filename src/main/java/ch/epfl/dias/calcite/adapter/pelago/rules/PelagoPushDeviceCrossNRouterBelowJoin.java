package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableMap;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;

public class PelagoPushDeviceCrossNRouterBelowJoin extends RelOptRule {
  public static final PelagoPushDeviceCrossNRouterBelowJoin INSTANCE = new PelagoPushDeviceCrossNRouterBelowJoin();

  protected PelagoPushDeviceCrossNRouterBelowJoin() {
    super(
      operand(
        PelagoDeviceCross.class,
        operand(
          PelagoRouter.class,
          operand(
            PelagoJoin.class,
            any()
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
    PelagoRouter      router = (PelagoRouter     ) call.rel(1);
    PelagoJoin        join   = (PelagoJoin       ) call.rel(2);
    RelNode           build  =                     join.getLeft ();
    RelNode           probe  =                     join.getRight();

    RelNode build_new = convert(
        convert(build, RelDeviceType.X86_64),
        RelHomDistribution.BRDCST
    );

    RelNode probe_new = convert(
        convert(probe, RelDeviceType.X86_64),
        RelHomDistribution.RANDOM
    );

    call.getPlanner().ensureRegistered(build_new, build);
    call.getPlanner().ensureRegistered(probe_new, probe);

    call.transformTo(
      join.copy(
        null,
        join.getCondition(),
//        PelagoDeviceCross.create(build, dcross.getDeviceType()),
//        PelagoDeviceCross.create(probe, dcross.getDeviceType()),,
        convert(
          build_new,
          build.getTraitSet()
            .replace(RelComputeDevice.from(dcross.getDeviceType()))
            .replace(dcross.getDeviceType())
        ),
        convert(
          probe_new,
          probe.getTraitSet()
            .replace(RelComputeDevice.from(dcross.getDeviceType()))
            .replace(dcross.getDeviceType())
        ),
        join.getJoinType(),
        join.isSemiJoinDone()
      )
    );
  }
}
