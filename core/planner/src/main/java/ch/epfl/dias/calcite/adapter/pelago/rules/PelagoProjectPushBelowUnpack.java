package ch.epfl.dias.calcite.adapter.pelago.rules;

import ch.epfl.dias.calcite.adapter.pelago.*;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.util.mapping.Mappings;

import com.google.common.collect.ImmutableMap;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

public class PelagoProjectPushBelowUnpack extends RelOptRule {
  public static final PelagoProjectPushBelowUnpack INSTANCE = new PelagoProjectPushBelowUnpack();

  protected PelagoProjectPushBelowUnpack() {
    super(operand(Project.class, operand(PelagoUnpack.class, operand(PelagoTableScan.class, any()))));
  }

  public void onMatch(RelOptRuleCall call) {
    Project           project = (Project          ) call.rel(0);
    PelagoUnpack      unpack  = (PelagoUnpack     ) call.rel(1);
    PelagoTableScan   scan    = (PelagoTableScan  ) call.rel(2);


    final Mappings.TargetMapping mapping = project.getMapping();

    SortedSet<Integer> projects2 = new TreeSet();
    for (RexNode x: project.getChildExps()){
      RelOptUtil.InputReferencedVisitor vis = new RelOptUtil.InputReferencedVisitor();
      x.accept(vis);
      projects2.addAll(vis.inputPosReferenced);
    }

    int[] scanfields = scan.fields();
    int[] fields = new int[projects2.size()]; //getProjectFields(project.getProjects(), scan);
    int[] revfields = new int[scan.getRowType().getFieldCount()];
    int i = 0;
    for (Integer j: projects2){//int i = 0; i < projects2.size(); i++) {
      fields[i] = scanfields[j];
      revfields[j] = i - j;
      i++;
    }

    RelOptUtil.RexInputConverter conv = new RelOptUtil.RexInputConverter(scan.getCluster().getRexBuilder(), scan.getRowType().getFieldList(), revfields);
    List<RexNode> projs = new ArrayList<RexNode>();
    boolean isId = true;
    for (int j = 0 ; j < project.getProjects().size() ; ++j) {
      projs.add(project.getProjects().get(j).accept(conv));
      RexNode p = project.getProjects().get(j).accept(conv);
      isId = isId && (p instanceof RexInputRef && ((RexInputRef) p).getIndex() == j);
    }

    var nscan =
        PelagoTableScan.create(
            scan.getCluster(),
            scan.getTable(),
            scan.pelagoTable(),
            fields
        );

    var nnscan = call.getPlanner().register(nscan, null);

    RelNode in = call.getPlanner().register(PelagoUnpack.create(nnscan, RelPacking.UnPckd), null);

    if (!isId){
      in = project.copy(
        project.getTraitSet(),
        in,
        projs,
        project.getRowType()
      );
    }
    call.transformTo(in);//, ImmutableMap.of(project, in));
  }

//  @Override public boolean matches(final RelOptRuleCall call) {
//    if (!super.matches(call)) return false;
//    PelagoTableScan   scan    = (PelagoTableScan  ) call.rel(2);
//    return scan.getPluginInfo().get("type") == "block";
//  }
}
