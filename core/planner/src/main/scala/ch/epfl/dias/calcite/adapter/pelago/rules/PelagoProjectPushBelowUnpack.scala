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

import ch.epfl.dias.calcite.adapter.pelago.rel.{PelagoTableScan, PelagoUnpack}
import ch.epfl.dias.calcite.adapter.pelago.traits.RelPacking
import org.apache.calcite.plan.RelOptRule.{any, operand}
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall, RelOptUtil}
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.{RexInputRef, RexNode}

import java.util
import scala.collection.JavaConverters._

object PelagoProjectPushBelowUnpack {
  val INSTANCE = new PelagoProjectPushBelowUnpack
}

class PelagoProjectPushBelowUnpack protected ()
    extends RelOptRule(
      operand(
        classOf[Project],
        operand(classOf[PelagoUnpack], operand(classOf[PelagoTableScan], any))
      )
    ) {
  override def onMatch(call: RelOptRuleCall): Unit = {
    val project = call.rel(0).asInstanceOf[Project]
    val unpack = call.rel(1).asInstanceOf[PelagoUnpack]
    val scan = call.rel(2).asInstanceOf[PelagoTableScan]
    val mapping = project.getMapping
    val projects2 = new util.TreeSet[Integer]
    for (x <- project.getProjects.asScala) {
      val vis = new RelOptUtil.InputReferencedVisitor
      x.accept(vis)
      projects2.addAll(vis.inputPosReferenced)
    }
    val scanfields = scan.fields
    val fields =
      new Array[Int](
        projects2.size
      ) //getProjectFields(project.getProjects(), scan);
    val revfields = new Array[Int](scan.getRowType.getFieldCount)
    var i = 0
    for (j <- projects2.asScala) { //int i = 0; i < projects2.size(); i++) {
      fields(i) = scanfields(j)
      revfields(j) = i - j
      i += 1
    }
    val conv = new RelOptUtil.RexInputConverter(
      scan.getCluster.getRexBuilder,
      scan.getRowType.getFieldList,
      revfields
    )
    val projs = new util.ArrayList[RexNode]
    var isId = true
    for (j <- 0 until project.getProjects.size) {
      projs.add(project.getProjects.get(j).accept(conv))
      val p = project.getProjects.get(j).accept(conv)
      isId = isId && (p
        .isInstanceOf[RexInputRef] && p.asInstanceOf[RexInputRef].getIndex == j)
    }
    val nscan = PelagoTableScan.create(
      scan.getCluster,
      scan.getTable,
      scan.pelagoTable,
      fields
    )
    val nnscan = call.getPlanner.register(nscan, null)
    var in = call.getPlanner
      .register(PelagoUnpack.create(nnscan, RelPacking.UnPckd), null)
    if (!isId)
      in = project.copy(project.getTraitSet, in, projs, project.getRowType)
    call.transformTo(in) //, ImmutableMap.of(project, in));

  }
//  @Override public boolean matches(final RelOptRuleCall call) {
//    if (!super.matches(call)) return false;
//    PelagoTableScan   scan    = (PelagoTableScan  ) call.rel(2);
//    return scan.getPluginInfo().get("type") == "block";
//  }
}
