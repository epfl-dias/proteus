package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.rules.ProjectMergeRule;
import org.apache.calcite.tools.RelBuilderFactory;
import org.apache.calcite.util.Permutation;

/**
 * This class is used to prevent triggering a cyclic dependency when two
 * projects are merged and the top one reverts the bottom one.
 */
public class PelagoProjectMergeRule extends ProjectMergeRule {
  public static final PelagoProjectMergeRule INSTANCE =
      new PelagoProjectMergeRule(true, RelFactories.LOGICAL_BUILDER);

  /**
   * Creates a PelagoProjectMergeRule, specifying whether to always merge projects.
   *
   * @param force Whether to always merge projects
   */
  protected PelagoProjectMergeRule(boolean force, RelBuilderFactory relBuilderFactory) {
    super(force, relBuilderFactory);
  }

  public void onMatch(RelOptRuleCall call) {
    final Project topProject = call.rel(0);
    final Project bottomProject = call.rel(1);

    // If one or both projects are permutations, short-circuit the complex logic
    // of building a RexProgram.
    final Permutation topPermutation = topProject.getPermutation();
    if (topPermutation != null) {
      if (topPermutation.isIdentity()) {
        // Let ProjectRemoveRule handle this.
        return;
      }
      final Permutation bottomPermutation = bottomProject.getPermutation();
      if (bottomPermutation != null) {
        if (bottomPermutation.isIdentity()) {
          // Let ProjectRemoveRule handle this.
          return;
        }
        final Permutation product = topPermutation.product(bottomPermutation);
        // If top projects reverts the bottom one, avoid creating a self-dependency
        if (product.isIdentity() && product.getTargetCount() == product.getSourceCount()){
          return;
        }
      }
    }

    super.onMatch(call);
  }
}
