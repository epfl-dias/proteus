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
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnnest;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

import java.util.Arrays;
import java.util.List;

public class PelagoPackTransfers extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    PelagoPackingConverterRule.TO_PACKED_INSTANCE,
    PelagoPackingConverterRule.TO_UNPCKD_INSTANCE,
//    PelagoDataLocalityConverterRule.INSTANCE,
//    new PelagoPushRouterDown(PelagoAggregate.class),
    new PelagoPackTransfers(PelagoRouter.class),
    new PelagoPackTransfers(PelagoDeviceCross.class),
//    new PelagoPushRouterDown(PelagoSort.class),
//    new PelagoPackTransfers(PelagoUnnest.class), //We only have a CPU-unnest for now
//    PelagoJoinPushBelowRouter.INSTANCE,
//    PelagoPushRouterBelowJoin.INSTANCE,
  };

//  private final Class op;

  protected PelagoPackTransfers(Class<? extends SingleRel> op) {
    super(operand(op, any()), "PPT" + op.getName());
//    this.op = op;
  }

  public void onMatch(RelOptRuleCall call) {
//    Pelago router  = call.rel(0);
    SingleRel    rel     = call.rel(0);
    RelNode      input   = rel.getInput();//              call.rel(2);

    call.transformTo(
      rel.copy(null, Arrays.asList(
        convert(
          input,
          RelPacking.Packed
        )
//        PelagoRouter.create(
//          convert(input, RelDeviceType.X86_64),
//          router.getDistribution()
//        )
      ))
    );
  }
}