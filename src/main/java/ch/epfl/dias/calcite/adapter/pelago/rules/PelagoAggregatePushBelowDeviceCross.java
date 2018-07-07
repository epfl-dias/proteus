package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;

import ch.epfl.dias.calcite.adapter.pelago.PelagoAggregate;
import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;

public class PelagoAggregatePushBelowDeviceCross extends RelOptRule {
  public static final PelagoAggregatePushBelowDeviceCross INSTANCE = new PelagoAggregatePushBelowDeviceCross();

  protected PelagoAggregatePushBelowDeviceCross() {
    super(operand(PelagoAggregate.class, operand(PelagoDeviceCross.class, any())));
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoAggregate   agg     = (PelagoAggregate  ) call.rel(0);
    PelagoDeviceCross decross = (PelagoDeviceCross) call.rel(1);

    PelagoAggregate new_agg = agg.copy(null, decross.getInput(), agg.indicator, agg.getGroupSet(), agg.getGroupSets(), agg.getAggCallList());
    call.transformTo(decross.copy(null, new_agg, decross.deviceType));


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
