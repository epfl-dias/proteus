package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

import java.util.stream.Collectors;

public class PelagoPushDeviceCrossDown extends RelOptRule {

  public static final RelOptRule[] RULES = {
    new PelagoPushDeviceCrossDown(PelagoAggregate.class),
    new PelagoPushDeviceCrossDown(PelagoFilter   .class),
    new PelagoPushDeviceCrossDown(PelagoProject  .class),
    new PelagoPushDeviceCrossDown(PelagoSort     .class),
    new PelagoPushDeviceCrossDown(PelagoJoin     .class),
  };

  protected PelagoPushDeviceCrossDown(Class<? extends RelNode> op) {
    super(operand(op, any()), "PPDCD" + op.getName());
  }

  protected RelNode cross(RelNode rel){
    return convert(rel, RelDeviceType.NVPTX);
  }

  public void onMatch(RelOptRuleCall call) {
    RelNode rel   = call.rel(0);

    var inps = rel.getInputs().stream().map(this::cross).collect(Collectors.toUnmodifiableList());

    call.transformTo(
        rel.copy(null, inps)
    );
  }
}
