package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

import java.util.stream.Collectors;

public class PelagoPackTransfers extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    new PelagoPackTransfers(PelagoUnion.class),
    new PelagoPackTransfers(PelagoRouter.class),
    new PelagoPackTransfers(PelagoSplit.class),
    new PelagoPackTransfers(PelagoDeviceCross.class)
  };

  protected PelagoPackTransfers(Class<? extends RelNode> op) {
    super(operand(op, any()), "PPT" + op.getName());
  }

  protected RelNode pack(RelNode rel){
    return convert(rel, RelPacking.Packed);
  }

  public void onMatch(RelOptRuleCall call) {
    RelNode rel   = call.rel(0);

    var inps = rel.getInputs().stream().map(this::pack).collect(Collectors.toUnmodifiableList());

    call.transformTo(
      rel.copy(null, inps)
    );
  }
}

