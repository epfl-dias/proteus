package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoJoin;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;

public class PelagoJoinPushBelowDeviceCross extends RelOptRule {
  public static final PelagoJoinPushBelowDeviceCross INSTANCE = new PelagoJoinPushBelowDeviceCross();

  protected PelagoJoinPushBelowDeviceCross() {
    super(
      operand(
        PelagoJoin.class,
        operand(PelagoDeviceCross.class,
          operand(RelNode.class, any())),
        operand(PelagoDeviceCross.class,
          operand(RelNode.class, any()))
      )
    );
  }

  public boolean matches(RelOptRuleCall call) {
    return true;
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoJoin        join          = (PelagoJoin       ) call.rel(0);
    PelagoDeviceCross left_decross  = ((PelagoDeviceCross) call.rel(1));
    PelagoDeviceCross right_decross = ((PelagoDeviceCross) call.rel(3));

    if (left_decross.getDeviceType() == right_decross.getDeviceType()) {
      RelNode lidecross = call.rel(2);
      RelNode ridecross = call.rel(4);

      PelagoJoin new_join = join.copy(null, join.getCondition(), lidecross, ridecross, join.getJoinType(), join.isSemiJoinDone());

      call.transformTo(PelagoDeviceCross.create(new_join, left_decross.getDeviceType()));
    }


//
//    Filter filter = (Filter)call.rel(0);
//    Project project = (Project)call.rel(1);
//    if (!RexOver.containsOver(project.getProjects(), (RexNode)null)) {
//      if (!RexUtil.containsCorrelation(filter.getCondition())) {
//        RexNode newCondition = RelOptUtil.pushPastProject(filter.getCondition(), project);
//        RelBuilder relBuilder = call.builder();
//        Object newFilterRel;
//        if (this.copyFilter) {
//          newFilterRel = filter.copy(filter.getTraitSet(), project.getInput(), RexUtil.removeNullabilityCast(relBuilder.getTypeFactory(), newCondition));
//        } else {
//          newFilterRel = relBuilder.push(project.getInput()).filter(new RexNode[]{newCondition}).build();
//        }
//
//        RelNode newProjRel = this.copyProject ? project.copy(project.getTraitSet(), (RelNode)newFilterRel, project.getProjects(), project.getRowType()) : relBuilder.push((RelNode)newFilterRel).project(project.getProjects(), project.getRowType().getFieldNames()).build();
//        call.transformTo((RelNode)newProjRel);
//      }
//    }
  }
}
