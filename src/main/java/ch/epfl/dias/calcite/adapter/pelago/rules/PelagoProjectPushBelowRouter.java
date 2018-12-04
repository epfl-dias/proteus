package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;

public class PelagoProjectPushBelowRouter extends RelOptRule {
  public static final PelagoProjectPushBelowRouter INSTANCE = new PelagoProjectPushBelowRouter();

  protected PelagoProjectPushBelowRouter() {
    super(operand(PelagoProject.class, operand(PelagoRouter.class, any())));
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoProject project = (PelagoProject) call.rel(0);
    PelagoRouter  router  = (PelagoRouter ) call.rel(1);

    PelagoProject new_project = project.copy(null, router.getInput(), project.getProjects(), project.getRowType());
    call.transformTo(router.copy(null, new_project, router.getDistribution()));


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
