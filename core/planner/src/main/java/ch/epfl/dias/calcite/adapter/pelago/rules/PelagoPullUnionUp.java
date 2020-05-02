package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;
import ch.epfl.dias.calcite.adapter.pelago.RelSplitPoint;

import java.util.stream.Collectors;

public class PelagoPullUnionUp extends RelOptRule {

  public static final RelOptRule[] RULES = new RelOptRule[]{
    new PelagoPullUnionUp(PelagoFilter.class),
    new PelagoPullUnionUp(PelagoProject.class),
    new PelagoPullUnionUp(PelagoAggregate.class),
    new RelOptRule(operand(PelagoJoin.class, operand(PelagoUnion.class, any()), operand(PelagoUnion.class, any())), "PPUUPelagoJoin") {
      @Override public void onMatch(final RelOptRuleCall call) {
        var rel = call.rel(0);
        var ins0 = call.rel(1).getInputs();
        var ins1 = call.rel(2).getInputs();

        call.transformTo(
          rel.copy(null,
            ImmutableList.of(
              convert(
                ins0.get(0),
                RelHetDistribution.SPLIT
              ),
              convert(
                ins1.get(0),
                RelHetDistribution.SPLIT_BRDCST
              )
            )
          )
        );

        call.transformTo(
          rel.copy(null,
            ImmutableList.of(
              convert(
                ins0.get(1),
                RelHetDistribution.SPLIT
              ),
              convert(
                ins1.get(1),
                RelHetDistribution.SPLIT_BRDCST
              )
            )
          )
        );
      }
    }
  };

  protected PelagoPullUnionUp(Class<? extends RelNode> op) {
    super(operand(op, operand(PelagoUnion.class, any())), "PPUU" + op.getName());
  }

  public void onMatch(RelOptRuleCall call) {
    var rel = call.rel(0);
    var rel2 = call.rel(1);

    if (rel instanceof PelagoAggregate && ((PelagoAggregate) rel).isGlobalAgg()) return;

    for (var e: rel2.getInputs()) {
      call.transformTo(
          rel.copy(null,
              ImmutableList.of(e)
          )
      );
    }
  }
}