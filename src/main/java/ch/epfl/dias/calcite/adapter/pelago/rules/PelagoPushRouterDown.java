package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnnest;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

import java.util.Arrays;
import java.util.List;

public class PelagoPushRouterDown extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    PelagoDistributionConverterRule.BRDCST_INSTANCE2,
    PelagoDistributionConverterRule.BRDCST_INSTANCE3,
    PelagoDistributionConverterRule.SEQNTL_INSTANCE2,
    PelagoDistributionConverterRule.RANDOM_INSTANCE ,
//    new PelagoPushRouterDown(PelagoAggregate.class),
    new PelagoPushRouterDown(PelagoFilter.class),
    new PelagoPushRouterDown(PelagoProject.class),
    new PelagoPushRouterDown(PelagoPack.class),
    new PelagoPushRouterDown(PelagoUnpack.class),
    new PelagoPushRouterDown(PelagoDeviceCross.class),
//    new PelagoPushRouterDown(PelagoSort.class),
//    new PelagoPushRouterDown(PelagoUnnest.class), //We only have a CPU-unnest for now
//    PelagoJoinPushBelowRouter.INSTANCE,
    PelagoPushRouterBelowJoin.INSTANCE,
    PelagoPushRouterBelowAggregate.INSTANCE
  };

//  private final Class op;

  protected PelagoPushRouterDown(Class<? extends SingleRel> op) {
    super(operand(PelagoRouter.class, operand(op, any())), "PPRD" + op.getName());
//    this.op = op;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoRouter router  = call.rel(0);
    SingleRel    rel     = call.rel(1);
    RelNode      input   = rel.getInput();//              call.rel(2);

    call.transformTo(
      rel.copy(null, Arrays.asList(
        convert(
          input,
          input.getTraitSet().replace(RelDeviceType.X86_64).replace(router.getDistribution())
        )
//        PelagoRouter.create(
//          convert(input, RelDeviceType.X86_64),
//          router.getDistribution()
//        )
      ))
    );
  }
}