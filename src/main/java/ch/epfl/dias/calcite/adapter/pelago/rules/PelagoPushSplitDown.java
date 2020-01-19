package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;

import java.util.stream.Collectors;

public class PelagoPushSplitDown extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    new PelagoPushSplitDown(PelagoFilter.class, RelHetDistribution.SPLIT),
    new PelagoPushSplitDown(PelagoFilter.class, RelHetDistribution.SPLIT_BRDCST),
    new PelagoPushSplitDown(PelagoProject.class, RelHetDistribution.SPLIT),
    new PelagoPushSplitDown(PelagoProject.class, RelHetDistribution.SPLIT_BRDCST),
    PelagoPushSplitBelowJoin.INSTANCE,
    PelagoPushSplitBelowAggregate.INSTANCE
  };

  final RelHetDistribution distr;

  protected PelagoPushSplitDown(Class<? extends RelNode> op,
      final RelHetDistribution distr) {
    super(operand(op, any()), "PPSD" + op.getName() + distr.toString());
    this.distr = distr;
  }

  protected static RelNode split(RelNode rel, RelHetDistribution distr){
    return convert(
      rel,
      rel.getTraitSet()
          .replace(distr)
          .replace(rel.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE) == RelDeviceType.NVPTX ? RelComputeDevice.NVPTX : RelComputeDevice.X86_64)
    );
  }

  protected RelNode split(RelNode rel){
    return split(rel, distr);
  }

  public void onMatch(RelOptRuleCall call) {
    var rel = call.rel(0);

    call.transformTo(
      rel.copy(null,
        rel.getInputs().stream().map(this::split).collect(Collectors.toList())
      )
    );
  }
}