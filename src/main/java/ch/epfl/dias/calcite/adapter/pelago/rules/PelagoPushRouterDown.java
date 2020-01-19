package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;

import java.util.stream.Collectors;

public class PelagoPushRouterDown extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    new PelagoPushRouterDown(PelagoFilter.class, RelHomDistribution.RANDOM),
    new PelagoPushRouterDown(PelagoProject.class, RelHomDistribution.RANDOM),
    new PelagoPushRouterDown(PelagoFilter.class, RelHomDistribution.BRDCST),
    new PelagoPushRouterDown(PelagoProject.class, RelHomDistribution.BRDCST),
    PelagoPushRouterBelowJoin.INSTANCE,
    PelagoPushRouterBelowAggregate.INSTANCE, //NoPartial
  };

  protected final RelHomDistribution trgt;

  protected PelagoPushRouterDown(Class<? extends RelNode> op,
      final RelHomDistribution trgt) {
    super(operand(op, any()), "PPRD" + op.getName() + trgt.toString());
//    this.op = op;
    this.trgt = trgt;
  }

  protected static RelNode route(RelNode input, RelHomDistribution trgt){
    return convert(
      input,
      input.getTraitSet().replace(trgt).replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)
    );
  }

  protected RelNode route(RelNode input){
    return route(input, trgt);
  }

  public boolean matches(RelOptRuleCall call){
    var trait = call.rel(0).getTraitSet();
    return trait.containsIfApplicable(RelHomDistribution.SINGLE);
  }

  public void onMatch(RelOptRuleCall call) {
    var rel = call.rel(0);

    call.transformTo(
      rel.copy(null,
        rel.getInputs().stream().map(this::route).collect(Collectors.toList())
      )
    );
  }
}