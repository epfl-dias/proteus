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
package ch.epfl.dias.calcite.adapter.pelago.metadata

import ch.epfl.dias.calcite.adapter.pelago.rel.{
  PelagoDeviceCross,
  PelagoPack,
  PelagoUnpack
}
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits.RelPacking
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.hep.HepRelVertex
import org.apache.calcite.rel.{BiRel, RelNode, SingleRel}
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core._
import org.apache.calcite.rel.metadata.{
  MetadataDef,
  MetadataHandler,
  ReflectiveRelMetadataProvider,
  RelMetadataProvider,
  RelMetadataQuery
}
import org.apache.calcite.rex.{RexLiteral, RexNode, RexProgram}

import java.util

object PelagoRelMdPacking {
  val SOURCE: RelMetadataProvider =
    ReflectiveRelMetadataProvider.reflectiveSource(
      Packing.method,
      new PelagoRelMdPacking
    )

  def table(table: RelOptTable): RelPacking = RelPacking.UnPckd
  def table(table: PelagoTable): RelPacking = table.getPacking

  /** Helper method to determine a
    * [[Sort]]'s deviceType. */
  def sort(mq: RelMetadataQuery, input: RelNode): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(input)

  /** Helper method to determine a
    * [[Filter]]'s deviceType. */
  def filter(mq: RelMetadataQuery, input: RelNode): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(input)

  /** Helper method to determine a
    * [[Aggregate]]'s deviceType. */
  def aggregate(mq: RelMetadataQuery, input: RelNode): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(input)

  /** Helper method to determine a
    * [[Exchange]]'s deviceType. */
  def exchange(mq: RelMetadataQuery, input: RelNode): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(input)

  /** Helper method to determine a
    * limit's deviceType. */
  def limit(mq: RelMetadataQuery, input: RelNode): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(input)

  /** Helper method to determine a
    * [[org.apache.calcite.rel.core.Calc]]'s deviceType. */
  def calc(mq: RelMetadataQuery, input: RelNode, program: RexProgram) =
    throw new AssertionError // TODO:
  /** Helper method to determine a [[Project]]'s collation. */
  def project(
      mq: RelMetadataQuery,
      input: RelNode,
      projects: util.List[_ <: RexNode]
  ): RelPacking = {
    val inputdeviceType = mq.asInstanceOf[PelagoRelMetadataQuery].packing(input)
//    final Mappings.TargetMapping mapping =
//        Project.getPartialMapping(input.getRowType().getFieldCount(),
//            projects);
    inputdeviceType //.apply(mapping); // TODO: Should we do something here ?

  }

  def values(
      rowType: RelDataType,
      tuples: ImmutableList[ImmutableList[RexLiteral]]
  ): RelPacking = RelPacking.UnPckd

  def devicecross(mq: RelMetadataQuery, input: RelNode): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(input)
  def unpack: RelPacking = RelPacking.UnPckd
  def pack: RelPacking = RelPacking.Packed
}

class PelagoRelMdPacking protected () extends MetadataHandler[Packing] {
  override def getDef: MetadataDef[Packing] = Packing.DEF

  /** Fallback method to deduce deviceType for any relational expression not
    * handled by a more specific method.
    *
    * @param rel Relational expression
    * @return Relational expression's deviceType
    */
  def packing(rel: RelNode, mq: RelMetadataQuery): RelPacking =
    RelPacking.UnPckd

  def packing(rel: SingleRel, mq: RelMetadataQuery): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(rel.getInput)

  def packing(rel: BiRel, mq: RelMetadataQuery): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(rel.getLeft)

  def packing(rel: SetOp, mq: RelMetadataQuery): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(rel.getInputs.get(0))

  def packing(scan: TableScan, mq: RelMetadataQuery): RelPacking =
    PelagoRelMdPacking.table(scan.getTable)

  def packing(project: Project, mq: RelMetadataQuery): RelPacking =
    PelagoRelMdPacking.project(mq, project.getInput, project.getProjects)

  def packing(values: Values, mq: RelMetadataQuery): RelPacking =
    PelagoRelMdPacking.values(values.getRowType, values.getTuples)

  def packing(
      devicecross: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): RelPacking =
    PelagoRelMdPacking.devicecross(mq, devicecross.getInput)

  def packing(rel: HepRelVertex, mq: RelMetadataQuery): RelPacking =
    mq.asInstanceOf[PelagoRelMetadataQuery].packing(rel.getCurrentRel)

  def packing(rel: PelagoPack, mq: RelMetadataQuery): RelPacking =
    PelagoRelMdPacking.pack

  def packing(rel: PelagoUnpack, mq: RelMetadataQuery): RelPacking =
    PelagoRelMdPacking.unpack
}
