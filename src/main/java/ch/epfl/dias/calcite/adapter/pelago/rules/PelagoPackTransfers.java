package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnnest;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

import java.util.Arrays;
import java.util.List;

public class PelagoPackTransfers extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    PelagoPackUnion.INSTANCE,
    new PelagoPackTransfers(PelagoRouter.class),
    new PelagoPackTransfers(PelagoSplit.class),
    new PelagoPackTransfers(PelagoDeviceCross.class)
  };

//  private final Class op;

  protected PelagoPackTransfers(Class<? extends SingleRel> op) {
    super(operand(op, any()), "PPT" + op.getName());
//    this.op = op;
  }

  public void onMatch(RelOptRuleCall call) {
//    Pelago router  = call.rel(0);
    SingleRel    rel     = call.rel(0);
//    RelNode      input   = call.rel(1);//              call.rel(2);

    call.transformTo(
      rel.copy(null, Arrays.asList(
        convert(
          rel.getInput(),
          RelPacking.Packed
        )
//        PelagoRouter.create(
//          convert(input, RelDeviceType.X86_64),
//          router.getDistribution()
//        )
      ))
    );
  }

  public static class PelagoPackUnion extends RelOptRule {
    public final static PelagoPackUnion INSTANCE = new PelagoPackUnion(PelagoUnion.class);

    protected PelagoPackUnion(Class<? extends RelNode> op) {
      super(operand(op, operand(PelagoUnpack.class, any()), operand(PelagoUnpack.class, any())), "PPS" + op.getName());
//    this.op = op;
    }

    public void onMatch(RelOptRuleCall call) {
      RelNode      rel   = call.rel(0);
      PelagoUnpack left  = call.rel(1);
      PelagoUnpack right = call.rel(2);

      call.transformTo(
        convert(
          rel.copy(
            null,
            ImmutableList.of(
              left.getInput(),
              right.getInput()
            )
          ),
          RelPacking.Packed
        )
      );
    }
  }
}

