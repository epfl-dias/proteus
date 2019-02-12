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
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;

import java.util.Arrays;

public class PelagoPushDeviceCrossNSplitDown extends RelOptRule {

  public static final RelOptRule[] RULES = {
//    PelagoDeviceTypeConverterRule.TO_NVPTX_INSTANCE ,
//    PelagoDeviceTypeConverterRule.TO_x86_64_INSTANCE,
//    new PelagoPushDeviceCrossNSplitDown(PelagoAggregate.class),
    new PelagoPushDeviceCrossNSplitDown(PelagoFilter   .class),
    new PelagoPushDeviceCrossNSplitDown(PelagoProject  .class),
    new PelagoPushDeviceCrossNSplitDown(PelagoPack     .class),
    new PelagoPushDeviceCrossNSplitDown(PelagoUnpack   .class),
    new PelagoPushDeviceCrossNSplitDown(PelagoSort     .class),
//    new PelagoPushDeviceCrossDown(PelagoUnnest   .class), //We only have a CPU-unnest for now
//    PelagoJoinPushBelowDeviceCross.INSTANCE,
    PelagoPushDeviceCrossNSplitBelowJoin.INSTANCE
  };

//  private final Class op;

  protected PelagoPushDeviceCrossNSplitDown(Class<? extends SingleRel> op) {
//    super(operand(PelagoDeviceCross.class, operand(op, operand(RelNode.class, any()))), "PPDCD" + op.getName());
    super(operand(PelagoDeviceCross.class, operand(PelagoSplit.class, operand(op, any()))), "PPDCSD" + op.getName());
//    this.op = op;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoDeviceCross decross = (PelagoDeviceCross) call.rel(0);
    PelagoSplit       split   = (PelagoSplit      ) call.rel(1);
    SingleRel         rel     = (SingleRel        ) call.rel(2);
    RelNode           inp     = rel.getInput();//                    call.rel(2);

    call.transformTo(
      rel.copy(null, Arrays.asList(
        convert(
          //        PelagoDeviceCross.create(
          inp,
          rel.getTraitSet().replace(split.hetdistribution()).replace(RelComputeDevice.from(decross.getDeviceType())).replace(decross.getDeviceType())
        )
      ))
    );
  }
}
