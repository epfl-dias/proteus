package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlSplittableAggFunction;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PelagoPushSplitBelowAggregate extends RelOptRule {
  public static final PelagoPushSplitBelowAggregate INSTANCE = new PelagoPushSplitBelowAggregate();

  protected PelagoPushSplitBelowAggregate() {
    super(operand(PelagoAggregate.class, any()), "PPSDAggregate");
  }

  public boolean matches(RelOptRuleCall call){
    PelagoAggregate rel    = call.rel(0);
    return rel.getTraitSet().containsIfApplicable(RelHomDistribution.SINGLE) &&
//        (rel.getGroupCount() == 0) &&
        (!rel.isGlobalAgg());
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

    RelOptPlanner planner = call.getPlanner();

    RelNode split = convert(rel.getInput(), RelHetDistribution.SPLIT);

    RelNode cpuSideAgg = planner.register(rel.copy(
        null,
        Arrays.asList(
          convert(
            convert(
                convert(
                    split,
                    RelHomDistribution.SINGLE //RANDOM_DISTRIBUTED //rel.getDistribution()
                ),
                RelDeviceType.X86_64
            ),
            RelComputeDevice.X86_64
          )
        )
    ), null);

    RelNode cpuSide = planner.register(convert(
        cpuSideAgg,
        cpuSideAgg.getTraitSet().replace(RelDeviceType.X86_64).replace(RelHomDistribution.SINGLE).replace(RelComputeDevice.X86_64).replace(RelHetDistribution.SPLIT)
    ), cpuSideAgg);

    RelNode gpuSideAgg = planner.register(rel.copy(
        null,
        Arrays.asList(
          convert(
            convert(
                convert(
                    split,
                    RelHomDistribution.SINGLE //RANDOM_DISTRIBUTED //rel.getDistribution()
                ),
                RelDeviceType.NVPTX
            ),
            RelComputeDevice.NVPTX
          )
        )
    ), cpuSide);

    RelNode gpuSide = planner.register(convert(
        gpuSideAgg,
        gpuSideAgg.getTraitSet().replace(RelDeviceType.X86_64).replace(RelHomDistribution.SINGLE).replace(RelComputeDevice.NVPTX).replace(RelHetDistribution.SPLIT)
    ), gpuSideAgg);

    RelNode union = PelagoUnion.create(ImmutableList.of(cpuSide, gpuSide), true);
    // Do not register the union as equivalent to the partial aggregates,
    // otherwise we loose the final aggregation.
    // So, is PelagoUnion a converter or not ?

    PelagoAggregate agg = PelagoAggregate.create(
        union,
        rel.indicator,
        topGroupSet,
        ImmutableList.of(topGroupSet),
        aggCalls,
        true
    );

    call.transformTo(agg);
  }
}
