package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.plan.RelOptUtil.RexInputConverter;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.mapping.Mappings;

import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoProject;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoTableScan;

import java.util.ArrayList;
import java.util.List;
import java.util.SortedSet;
import java.util.TreeSet;

/**
 * Planner rule that projects from a {@link PelagoTableScan} scan just the
 * columns needed to satisfy a projection. If the projection's expressions are
 * trivial, the projection is removed.
 *
 * Based on:
 * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTableScanRule.java
 */
public class PelagoProjectTableScanRule extends RelOptRule {
  public static final PelagoProjectTableScanRule INSTANCE =
      new PelagoProjectTableScanRule(PelagoRelFactories.PELAGO_BUILDER);

  /**
   * Creates a PelagoProjectTableScanRule.
   *
   * @param relBuilderFactory Builder for relational expressions
   */
  public PelagoProjectTableScanRule(RelBuilderFactory relBuilderFactory) {
    super(
        operand(Project.class,
            operand(PelagoTableScan.class, none())),
        relBuilderFactory,
        "PelagoProjectTableScanRule");
  }

  @Override public void onMatch(RelOptRuleCall call) {
    final Project project = call.rel(0);
    final PelagoTableScan scan = call.rel(1);

    call.transformTo(merge(project, scan));
  }

  public static RelNode merge(Project project, PelagoTableScan scan) {
    final Mappings.TargetMapping mapping = project.getMapping();

    SortedSet<Integer> projects2 = new TreeSet();
    for (RexNode x: project.getProjects()){
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

    RexInputConverter conv = new RexInputConverter(scan.getCluster().getRexBuilder(), scan.getRowType().getFieldList(), revfields);
    List<RexNode> projs = new ArrayList<RexNode>();
    for (int j = 0 ; j < project.getProjects().size() ; ++j) {
      projs.add(project.getProjects().get(j).accept(conv));
    }

    return PelagoProject.create(
        PelagoTableScan.create(
            scan.getCluster(),
            scan.getTable(),
            scan.pelagoTable(),
            fields),
        projs,
        project.getRowType(),
        project.getHints()
    );
//
//    call.transformTo(
//        PelagoTableScan.create(
//            scan.getCluster(),
//            scan.getTable(),
//            scan.pelagoTable(),
//            fields));
  }

//  private int[] getProjectFields(List<RexNode> exps, PelagoTableScan scan) {
//    final int[] fields = new int[exps.size()];
//    for (int i = 0; i < exps.size(); i++) {
//      final RexNode exp = exps.get(i);
//      if (exp instanceof RexInputRef) {
//        if (scan.fields() != null){
//          fields[i] = scan.fields()[((RexInputRef) exp).getIndex()];
//        } else {
//          fields[i] = ((RexInputRef) exp).getIndex();
//        }
//      } else {
//        return null; // not a simple projection
//      }
//    }
//    return fields;
//  }
}

// End PelagoProjectTableScanRule.java
