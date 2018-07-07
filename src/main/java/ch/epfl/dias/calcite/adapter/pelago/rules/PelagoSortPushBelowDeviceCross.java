package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.sql.SqlExplainLevel;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSort;

public class PelagoSortPushBelowDeviceCross extends RelOptRule {
  public static final PelagoSortPushBelowDeviceCross INSTANCE = new PelagoSortPushBelowDeviceCross();

  protected PelagoSortPushBelowDeviceCross() {
    super(operand(PelagoSort.class, operand(PelagoDeviceCross.class, any())));
  }

  public void onMatch(RelOptRuleCall call) {
    PelagoSort        sort    = (PelagoSort       ) call.rel(0);
    PelagoDeviceCross decross = (PelagoDeviceCross) call.rel(1);

    PelagoSort new_sort = PelagoSort.create(decross.getInput(), sort.collation, sort.fetch, sort.offset);

    call.transformTo(decross.copy(null, new_sort, decross.deviceType));


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
