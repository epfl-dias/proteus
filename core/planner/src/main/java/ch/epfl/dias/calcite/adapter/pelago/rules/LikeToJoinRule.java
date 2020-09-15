package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.volcano.VolcanoPlanner;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.NlsString;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDictTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelBuilder;

import java.util.Set;

public class LikeToJoinRule extends RelOptRule {
  public static final LikeToJoinRule INSTANCE = new LikeToJoinRule();


  protected LikeToJoinRule() {
    super(
        operand(
            Filter.class,
            any()
        )
    );
  }

  class FindLikes extends RexShuttle {
    final RexBuilder builder;
    final RelBuilder relBuilder;
    int cnt;
    RelNode input;

    public FindLikes(RexBuilder builder, RelNode input){
      super();
      this.builder = builder;
      this.cnt = input.getRowType().getFieldCount();
      this.input = input;
      this.relBuilder = PelagoRelBuilder.create(Frameworks.newConfigBuilder().build());
    }

    public RexNode visitCall(RexCall call){
      if (call.getKind() == SqlKind.LIKE) {
        System.out.println("Found one!");
//
//        input.getTable().getRelOptSchema()

        String name = "." + ((RexInputRef) call.getOperands().get(0)).getName() + ".dict";
        System.out.println(name);
//        System.out.println(input.getTable().getQualifiedName());

        Set<RexNode> ref = input.getCluster().getMetadataQuery().getExpressionLineage(input, call.getOperands().get(0));
        assert(ref != null) : "Have you forgot to add an operator in the expression lineage metadata provider?";
        assert(ref.size() == 1);
        System.out.println(ref.iterator().next());

        // NOTE: ok! that's a good sign!
        int attrIndex = ((RexTableInputRef) ref.iterator().next()).getIndex();
        String regex = ((NlsString) ((RexLiteral) call.getOperands().get(1)).getValue()).getValue();
        RelOptTable table = ((RexTableInputRef) ref.iterator().next()).getTableRef().getTable();
        System.out.println(table);
        System.out.println();
        table.getRelOptSchema().getTableForMember(table.getQualifiedName());

        input = relBuilder
          .push(input)
          .push(PelagoDictTableScan.create(input.getCluster(), table, regex, attrIndex))
          .join(JoinRelType.INNER, builder.makeLiteral(true))
          .build();

        return builder.makeCall(
          SqlStdOperatorTable.EQUALS,
          call.getOperands().get(0),
          builder.makeInputRef(call.getOperands().get(0).getType(), cnt++)
        );
      }
      return super.visitCall(call);
    }

    public RelNode getNewInput(){
      return input;
    }
  }

  @Override
  public void onMatch(final RelOptRuleCall call) {
    Filter filter = call.rel(0);
    RexNode cond = filter.getCondition();

    RexBuilder rexBuilder = filter.getCluster().getRexBuilder();
    FindLikes fl = new FindLikes(rexBuilder, filter.getInput());
    RexNode new_cond = cond.accept(fl);

    if (fl.getNewInput() != filter.getInput()){
      // do not consider the matched node again!
      call.getPlanner().prune(filter);

      ImmutableList.Builder<RexNode> projs = ImmutableList.<RexNode>builder();
      for (RelDataTypeField f: filter.getRowType().getFieldList()){
        projs.add(rexBuilder.makeInputRef(f.getType(), f.getIndex()));
      }

      RelNode replacement = call.builder()
          .push(fl.getNewInput())
          .filter(new_cond)
          .project(projs.build())
          .build();

      System.out.println(RelOptUtil.toString(replacement, SqlExplainLevel.ALL_ATTRIBUTES));

      // push transformation
      call.transformTo(replacement);
    }
  }
}
