package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

import java.util.Arrays;

public class PelagoPushDeviceCrossNRouterDown extends RelOptRule {

  public static final RelOptRule[] RULES = {
////    PelagoDeviceTypeConverterRule.TO_NVPTX_INSTANCE ,
////    PelagoDeviceTypeConverterRule.TO_x86_64_INSTANCE,
////    new PelagoPushDeviceCrossNSplitDown(PelagoAggregate.class),
//    new PelagoPushDeviceCrossNRouterDown(PelagoFilter   .class),
//    new PelagoPushDeviceCrossNRouterDown(PelagoProject  .class),
//    new PelagoPushDeviceCrossNRouterDown(PelagoPack     .class),
//    new PelagoPushDeviceCrossNRouterDown(PelagoUnpack   .class),
//    new PelagoPushDeviceCrossNRouterDown(PelagoSort     .class),
////    new PelagoPushDeviceCrossDown(PelagoUnnest   .class), //We only have a CPU-unnest for now
////    PelagoJoinPushBelowDeviceCross.INSTANCE,
//    PelagoPushDeviceCrossNSplitBelowJoin.INSTANCE
  };

//  private final Class op;

  protected PelagoPushDeviceCrossNRouterDown(Class<? extends SingleRel> op) {
//    super(operand(PelagoDeviceCross.class, operand(op, operand(RelNode.class, any()))), "PPDCD" + op.getName());
    super(operand(PelagoDeviceCross.class, operand(PelagoRouter.class, operand(op, any()))), "PPDCSRD" + op.getName());
//    this.op = op;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoDeviceCross decross = (PelagoDeviceCross) call.rel(0);
    PelagoRouter      router  = (PelagoRouter     ) call.rel(1);
    SingleRel         rel     = (SingleRel        ) call.rel(2);
    RelNode           inp     = rel.getInput();//                    call.rel(2);

    RelNode inp_new = convert(
        convert(inp, RelDeviceType.X86_64),
        router.getHomDistribution()
    );

    call.getPlanner().ensureRegistered(inp_new, inp);

    call.transformTo(
      rel.copy(null, Arrays.asList(
        convert(
          inp_new,
          inp.getTraitSet()
            .replace(RelComputeDevice.from(decross.getDeviceType()))
            .replace(decross.getDeviceType())
        )
      ))
    );
  }
}
