package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

import java.util.Arrays;

public class PelagoPushSplitDown extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
//    PelagoDistributionConverterRule.BRDCST_INSTANCE2,
//    PelagoDistributionConverterRule.BRDCST_INSTANCE3,
//    PelagoDistributionConverterRule.BRDCST_INSTANCE4,
//    PelagoDistributionConverterRule.SEQNTL_INSTANCE2,
//    PelagoDistributionConverterRule.RANDOM_INSTANCE ,
//    PelagoDistributionConverterRule.RANDOM_INSTANCE2,
    PelagoHetDistributionConverterRule.RANDOM_INSTANCE,
    PelagoHetDistributionConverterRule.BRDCST_INSTANCE,
//    new PelagoPushRouterDown(PelagoAggregate.class),
    new PelagoPushSplitDown(PelagoFilter.class),
    new PelagoPushSplitDown(PelagoProject.class),
    new PelagoPushSplitDown(PelagoPack.class),
    new PelagoPushSplitDown(PelagoUnpack.class),
    new PelagoPushSplitDown(PelagoDeviceCross.class),
//    new PelagoPushRouterDown(PelagoSort.class),
//    new PelagoPushRouterDown(PelagoUnnest.class), //We only have a CPU-unnest for now
//    PelagoJoinPushBelowRouter.INSTANCE,
    PelagoPushSplitBelowJoin.INSTANCE,
    PelagoPushSplitBelowAggregate.INSTANCE
  };

//  private final Class op;

  protected PelagoPushSplitDown(Class<? extends SingleRel> op) {
    super(operand(PelagoSplit.class, operand(op, any())), "PPSD" + op.getName());
//    this.op = op;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoSplit  split = call.rel(0);
    SingleRel    rel   = call.rel(1);
    RelNode      input = rel.getInput();//              call.rel(2);

    call.transformTo(
      rel.copy(null, Arrays.asList(
        convert(
          input,
          input.getTraitSet().replace(split.hetdistribution())
        )
//        PelagoRouter.create(
//          convert(input, RelDeviceType.X86_64),
//          router.getDistribution()
//        )
      ))
    );
  }
}