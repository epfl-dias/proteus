package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoFilter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;

public class PelagoFilterPushBelowDeviceCross extends RelOptRule {
  public static final PelagoFilterPushBelowDeviceCross INSTANCE = new PelagoFilterPushBelowDeviceCross();

  protected PelagoFilterPushBelowDeviceCross() {
    super(operand(PelagoFilter.class, operand(PelagoDeviceCross.class, any())));
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoFilter      filter  = (PelagoFilter     ) call.rel(0);
    PelagoDeviceCross decross = (PelagoDeviceCross) call.rel(1);

    PelagoFilter new_filter = filter.create(decross.getInput(), filter.getCondition());
    call.transformTo(decross.copy(null, new_filter, decross.getDeviceType()));


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
