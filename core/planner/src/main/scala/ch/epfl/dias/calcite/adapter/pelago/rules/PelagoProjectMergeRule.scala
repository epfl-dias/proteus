/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
 */

package ch.epfl.dias.calcite.adapter.pelago.rules

import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.rel.core.{Project, RelFactories}
import org.apache.calcite.rel.rules.ProjectMergeRule
import org.apache.calcite.tools.RelBuilderFactory

/**
  * This class is used to prevent triggering a cyclic dependency when two
  * projects are merged and the top one reverts the bottom one.
  */
object PelagoProjectMergeRule {
  val INSTANCE = new PelagoProjectMergeRule(true, RelFactories.LOGICAL_BUILDER)
}

/**
  * Creates a PelagoProjectMergeRule, specifying whether to always merge projects.
  *
  * @param force Whether to always merge projects
  */
class PelagoProjectMergeRule protected (
    val force: Boolean,
    relBuilderFactory: RelBuilderFactory
) extends ProjectMergeRule(force, relBuilderFactory) {
  override def onMatch(call: RelOptRuleCall): Unit = {
    val topProject: Project = call.rel(0)
    val bottomProject: Project = call.rel(1)
// If one or both projects are permutations, short-circuit the complex logic
// of building a RexProgram.
    val topPermutation = topProject.getPermutation
    if (topPermutation != null) {
      if (topPermutation.isIdentity) { // Let ProjectRemoveRule handle this.
        return
      }
      val bottomPermutation = bottomProject.getPermutation
      if (bottomPermutation != null) {
        if (bottomPermutation.isIdentity) return
        val product = topPermutation.product(bottomPermutation)
// If top projects reverts the bottom one, avoid creating a self-dependency
        if (
          product.isIdentity && product.getTargetCount == product.getSourceCount
        ) return
      }
    }
    super.onMatch(call)
  }
}
