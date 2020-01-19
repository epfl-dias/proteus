package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlSplittableAggFunction;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;

import java.util.ArrayList;
import java.util.List;

public class PelagoPartialAggregateRule extends RelOptRule {
  public static final PelagoPartialAggregateRule INSTANCE = new PelagoPartialAggregateRule();

  protected PelagoPartialAggregateRule() {
    super(operand(PelagoAggregate.class, any()), "PelagoPartialAggregate");
  }

  public boolean matches(RelOptRuleCall call){
    PelagoAggregate rel    = call.rel(0);
    return rel.getTraitSet().containsIfApplicable(RelHomDistribution.SINGLE) &&
//        (rel.getGroupCount() == 0) &&
        (rel.isGlobalAgg() && !rel.isSplitted());
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoAggregate rel    = call.rel(0);

    List<AggregateCall> aggCalls = new ArrayList<AggregateCall>();

    RexBuilder rexBuilder = rel.getCluster().getRexBuilder();

    Integer i = rel.getGroupCount();
    for (AggregateCall a: rel.getAggCallList()){
      SqlSplittableAggFunction s = a.getAggregation().unwrap(SqlSplittableAggFunction.class);

      List<RexNode> list = new ArrayList<>(rexBuilder.identityProjects(rel.getRowType()));
      SqlSplittableAggFunction.Registry<RexNode> reg = e -> {
        int i1 = list.indexOf(e);
        if (i1 < 0) {
          i1 = list.size();
          list.add(e);
        }
        return i1;
      };

      AggregateCall aTop = s.topSplit(
          rexBuilder,
        reg,
        rel.getGroupCount(),
        rel.getRowType(),
        a,
        i,
        -1
        );

//      if (a.getAggregation().getKind() == SqlKind.COUNT){
//        aggCalls.add(new AggregateCall(SqlSplittableAggFunction.CountSplitter.INSTANCE, AggFunction.SUM, a.isDistinct(), List.of(i), a.getType(), a.name));
//      } else {
//        aggCalls.add(new AggregateCall(a.getAggregation(), a.isDistinct(), List.of(i), a.getType(), a.name));
//      }
      aggCalls.add(aTop);
      i = i + 1;
    }

    ImmutableBitSet topGroupSet = ImmutableBitSet.builder().set(0, rel.getGroupCount()).build();

    var locagg = rel.copy(
        rel.getInput(),
        false, true
    );

    call.transformTo(
      call.getPlanner().register(
        PelagoAggregate.create(
            locagg,
            topGroupSet,
            ImmutableList.of(topGroupSet),
            aggCalls,
            true, true
        ),
        rel
      )
    );
  }
}
