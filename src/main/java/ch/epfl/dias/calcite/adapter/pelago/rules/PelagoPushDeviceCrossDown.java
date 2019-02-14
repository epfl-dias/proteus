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
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;

import java.util.Arrays;
import java.util.List;

public class PelagoPushDeviceCrossDown extends RelOptRule {

  public static final RelOptRule[] RULES = {
//    PelagoDeviceTypeConverterRule.TO_NVPTX_INSTANCE ,
//    PelagoDeviceTypeConverterRule.TO_x86_64_INSTANCE,
    new PelagoPushDeviceCrossDown(PelagoAggregate.class),
    new PelagoPushDeviceCrossDown(PelagoFilter   .class),
    new PelagoPushDeviceCrossDown(PelagoProject  .class),
    new PelagoPushDeviceCrossDown(PelagoPack     .class),
    new PelagoPushDeviceCrossDown(PelagoUnpack   .class),
    new PelagoPushDeviceCrossDown(PelagoSort     .class),
//    new PelagoPushDeviceCrossDown(PelagoUnnest   .class), //We only have a CPU-unnest for now
//    PelagoJoinPushBelowDeviceCross.INSTANCE,
    PelagoPushDeviceCrossBelowJoin.INSTANCE
  };

//  private final Class op;

  protected PelagoPushDeviceCrossDown(Class<? extends SingleRel> op) {
//    super(operand(PelagoDeviceCross.class, operand(op, operand(RelNode.class, any()))), "PPDCD" + op.getName());
    super(operand(PelagoDeviceCross.class, operand(op, any())), "PPDCD" + op.getName());
//    this.op = op;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoDeviceCross decross = (PelagoDeviceCross) call.rel(0);
    SingleRel         rel     = (SingleRel        ) call.rel(1);
    RelNode           inp     = rel.getInput();//                    call.rel(2);

    call.transformTo(
      rel.copy(null, Arrays.asList(
        convert(
//        PelagoDeviceCross.create(
          inp,
          decross.getDeviceType()
        )
      ))
    );
  }
}
