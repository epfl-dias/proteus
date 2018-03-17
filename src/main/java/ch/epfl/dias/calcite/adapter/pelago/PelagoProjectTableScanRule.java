package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.tools.RelBuilderFactory;

import java.util.List;

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
      new PelagoProjectTableScanRule(RelFactories.LOGICAL_BUILDER);

  /**
   * Creates a PelagoProjectTableScanRule.
   *
   * @param relBuilderFactory Builder for relational expressions
   */
  public PelagoProjectTableScanRule(RelBuilderFactory relBuilderFactory) {
    super(
        operand(LogicalProject.class,
            operand(PelagoTableScan.class, none())),
        relBuilderFactory,
        "PelagoProjectTableScanRule");
  }

  @Override public void onMatch(RelOptRuleCall call) {
    final LogicalProject project = call.rel(0);
    final PelagoTableScan scan = call.rel(1);
    int[] fields = getProjectFields(project.getProjects(), scan);
    if (fields == null) {
      // Project contains expressions more complex than just field references.
      return;
    }
    call.transformTo(
        new PelagoTableScan(
            scan.getCluster(),
            scan.getTable(),
            scan.pelagoTable,
            fields));
  }

  private int[] getProjectFields(List<RexNode> exps, PelagoTableScan scan) {
    final int[] fields = new int[exps.size()];
    for (int i = 0; i < exps.size(); i++) {
      final RexNode exp = exps.get(i);
      if (exp instanceof RexInputRef) {
        if (scan.fields != null){
          fields[i] = scan.fields[((RexInputRef) exp).getIndex()];
        } else {
          fields[i] = ((RexInputRef) exp).getIndex();
        }
      } else {
        return null; // not a simple projection
      }
    }
    return fields;
  }
}

// End CsvProjectTableScanRule.java
