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

import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories
import ch.epfl.dias.calcite.adapter.pelago.rel.{PelagoProject, PelagoTableScan}
import org.apache.calcite.plan.RelOptRule.{none, operand}
import org.apache.calcite.plan.{RelOptRule, RelOptRuleCall, RelOptUtil}
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rex.RexNode
import org.apache.calcite.tools.RelBuilderFactory

import java.util
import scala.collection.JavaConverters._

/**
  * Planner rule that projects from a [[PelagoTableScan]] scan just the
  * columns needed to satisfy a projection. If the projection's expressions are
  * trivial, the projection is removed.
  *
  * Based on:
  * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTableScanRule.java
  */
object PelagoProjectTableScanRule {
  val INSTANCE = new PelagoProjectTableScanRule(
    PelagoRelFactories.PELAGO_BUILDER
  )
  def merge(project: Project, scan: PelagoTableScan): PelagoProject = {
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
    for (j <- 0 until project.getProjects.size) {
      projs.add(project.getProjects.get(j).accept(conv))
    }
    PelagoProject.create(
      PelagoTableScan
        .create(scan.getCluster, scan.getTable, scan.pelagoTable, fields),
      projs,
      project.getRowType,
      project.getHints
    )
//
//    call.transformTo(
//        PelagoTableScan.create(
//            scan.getCluster(),
//            scan.getTable(),
//            scan.pelagoTable(),
//            fields));
  }
}

/**
  * Creates a PelagoProjectTableScanRule.
  *
  * @param relBuilderFactory Builder for relational expressions
  */
class PelagoProjectTableScanRule(relBuilderFactory: RelBuilderFactory)
    extends RelOptRule(
      operand(classOf[Project], operand(classOf[PelagoTableScan], none)),
      relBuilderFactory,
      "PelagoProjectTableScanRule"
    ) {
  override def onMatch(call: RelOptRuleCall): Unit = {
    val project: Project = call.rel(0)
    val scan: PelagoTableScan = call.rel(1)
    call.transformTo(PelagoProjectTableScanRule.merge(project, scan))
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
