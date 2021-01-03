package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHetDistribution;

public class PelagoPushSplitBelowJoin extends PelagoPushSplitDown {
  public static final PelagoPushSplitBelowJoin INSTANCE = new PelagoPushSplitBelowJoin();

  protected PelagoPushSplitBelowJoin() {
    super(
      PelagoJoin.class,
      RelHetDistribution.SPLIT
    );
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoJoin join = call.rel(0);
    RelNode build = join.getLeft();
    RelNode probe = join.getRight();

    RelNode new_build = split(
      build,
      RelHetDistribution.SPLIT_BRDCST
    );

    RelNode new_probe = split(
      probe
    );

    call.transformTo(
      join.copy(
        null,
        ImmutableList.of(new_build, new_probe)
      )
    );
  }
}
