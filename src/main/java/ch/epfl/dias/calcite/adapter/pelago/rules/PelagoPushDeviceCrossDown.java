package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

import java.util.stream.Collectors;

public class PelagoPushDeviceCrossDown extends ConverterRule {

  public static final RelOptRule[] RULES = {
    new PelagoPushDeviceCrossDown(PelagoAggregate.class),
    new PelagoPushDeviceCrossDown(PelagoFilter   .class),
    new PelagoPushDeviceCrossDown(PelagoProject  .class),
    new PelagoPushDeviceCrossDown(PelagoSort     .class),
    new PelagoPushDeviceCrossDown(PelagoJoin     .class),
  };

  protected PelagoPushDeviceCrossDown(Class<? extends RelNode> op) {
    super(op, RelDeviceType.X86_64, RelDeviceType.NVPTX, "PPDCD" + op.getName());
  }

  protected RelNode cross(RelNode rel){
    return convert(rel, RelDeviceType.NVPTX);
  }

  public RelNode convert(RelNode rel) {
    var inps = rel.getInputs().stream().map(this::cross).collect(Collectors.toUnmodifiableList());

    return rel.copy(null, inps);
  }
}
