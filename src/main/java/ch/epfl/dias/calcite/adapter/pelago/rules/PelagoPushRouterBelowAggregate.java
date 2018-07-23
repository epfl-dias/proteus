package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlSplittableAggFunction;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PelagoPushRouterBelowAggregate extends RelOptRule {
  public static final PelagoPushRouterBelowAggregate INSTANCE = new PelagoPushRouterBelowAggregate();

  protected PelagoPushRouterBelowAggregate() {
    super(operand(PelagoAggregate.class, any()), "PPRDAggregate");
  }

  public boolean matches(RelOptRuleCall call){
    PelagoAggregate rel    = call.rel(0);
    return rel.getTraitSet().containsIfApplicable(RelDistributions.SINGLETON) &&
//        (rel.getGroupCount() == 0) &&
        (!rel.isGlobalAgg());
  }

  public void onMatch(RelOptRuleCall call) {
//    PelagoRouter    router = call.rel(0);
    PelagoAggregate rel    = call.rel(0);

//    call.getPlanner().setImportance(rel, 0);

//    rel.getGroupSet().

    List<AggregateCall> aggCalls = new ArrayList<AggregateCall>();

    RexBuilder rexBuilder = rel.getCluster().getRexBuilder();

    Integer i = rel.getGroupCount();
    for (AggregateCall a: rel.getAggCallList()){
      SqlSplittableAggFunction s = a.getAggregation().unwrap(SqlSplittableAggFunction.class);

      List<RexNode> list = new ArrayList<>(rexBuilder.identityProjects(rel.getRowType()));
      SqlSplittableAggFunction.Registry<RexNode> reg = new SqlSplittableAggFunction.Registry<RexNode>(){
        public int register(RexNode e){
          int i = list.indexOf(e);
          if (i < 0) {
            i = list.size();
            list.add(e);
          }
          return i;
        }
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

    call.transformTo(
      PelagoAggregate.create(
          PelagoRouter.create(
              convert(
                  rel.copy(
                      null,
                      Arrays.asList(
                          convert(
                              convert(
                                  rel.getInput(),
                                  RelDistributions.RANDOM_DISTRIBUTED //rel.getDistribution()
                              ),
                              rel.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)
                          )
                      )
                  ),
                  RelDeviceType.X86_64
              ),
              RelDistributions.SINGLETON
          ),
          rel.indicator,
          topGroupSet,
          ImmutableList.of(topGroupSet),
          aggCalls,
          true
      )
    );



//    call.transformTo(
//        rel.copy(null, Arrays.asList(
//            convert(
//                input,
//                input.getTraitSet().replace(RelDeviceType.X86_64).replace(router.getDistribution())
//            )
////        PelagoRouter.create(
////          convert(input, RelDeviceType.X86_64),
////          router.getDistribution()
////        )
//        ))
//    );
  }
}
