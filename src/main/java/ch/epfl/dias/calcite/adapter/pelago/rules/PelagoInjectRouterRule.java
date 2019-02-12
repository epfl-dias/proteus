package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.Exchange;
import org.apache.calcite.rel.metadata.RelMdDistribution;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;
import ch.epfl.dias.calcite.adapter.pelago.PelagoToEnumerableConverter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnnest;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

import java.util.Arrays;
import java.util.List;

public class PelagoInjectRouterRule extends RelOptRule {

  public static final PelagoInjectRouterRule INSTANCE = new PelagoInjectRouterRule();

//  private final Class op;

  protected PelagoInjectRouterRule() {
    super(operand(PelagoToEnumerableConverter.class, operand(RelNode.class, any())), "PelagoInjectRouterRule");
//    this.op = op;
  }

  public boolean matches(RelOptRuleCall call) {
    RelNode rel = call.rel(1);
    return !(rel instanceof Exchange) && rel.getTraitSet().containsIfApplicable(RelDistributions.SINGLETON);
  }


  public void onMatch(RelOptRuleCall call) {
    PelagoToEnumerableConverter con = call.rel(0);
    RelNode                     rel = call.rel(1);

    call.transformTo(
      con.copy(null,
        PelagoRouter.create(
          convert(rel, RelDistributions.RANDOM_DISTRIBUTED),
          RelDistributions.SINGLETON
        )
      )
    );
  }
}