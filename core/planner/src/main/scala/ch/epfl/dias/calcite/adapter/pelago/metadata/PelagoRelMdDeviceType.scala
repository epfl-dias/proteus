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
  PelagoTableScan,
  PelagoToEnumerableConverter
}
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelDeviceType,
  RelDeviceTypeTraitDef
}
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.plan.hep.HepRelVertex
import org.apache.calcite.rel.{BiRel, RelNode, SingleRel}
import org.apache.calcite.rel.core._
import org.apache.calcite.rel.metadata.{
  MetadataDef,
  MetadataHandler,
  ReflectiveRelMetadataProvider,
  RelMetadataProvider,
  RelMetadataQuery
}
import org.apache.calcite.rel.rules.MultiJoin
import org.apache.calcite.rex.RexNode

import java.util

object PelagoRelMdDeviceType {
  private val INSTANCE = new PelagoRelMdDeviceType
  val SOURCE: RelMetadataProvider =
    ReflectiveRelMetadataProvider.reflectiveSource(
      DeviceType.method,
      PelagoRelMdDeviceType.INSTANCE
    )

  /** Helper method to determine a
    * [[Aggregate]]'s deviceType. */
  def aggregate(mq: RelMetadataQuery, input: RelNode): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(input)

  /** Helper method to determine a
    * [[Filter]]'s deviceType. */
  def filter(mq: RelMetadataQuery, input: RelNode): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(input)

  /** Helper method to determine a [[Project]]'s collation. */
  def project(
      mq: RelMetadataQuery,
      input: RelNode,
      projects: util.List[_ <: RexNode]
  ): RelDeviceType = {
    val inputdeviceType =
      mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(input)
//    final Mappings.TargetMapping mapping =
//        Project.getPartialMapping(input.getRowType().getFieldCount(),
//            projects);
    inputdeviceType //.apply(mapping); // TODO: Should we do something here ?

  }

  /** Helper method to determine a
    * [[Sort]]'s deviceType. */ // Helper methods
  def sort(mq: RelMetadataQuery, input: RelNode): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(input)

  /** Helper method to determine a
    * [[TableScan]]'s deviceType. */
  def table(table: RelOptTable): RelDeviceType = RelDeviceType.X86_64
  def table(table: PelagoTable): RelDeviceType = table.getDeviceType

  def pelagoToEnumberable(mq: RelMetadataQuery, input: RelNode): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(input)
}

class PelagoRelMdDeviceType extends MetadataHandler[DeviceType] {
  override def getDef: MetadataDef[DeviceType] = DeviceType.DEF

  def deviceType(scan: TableScan, mq: RelMetadataQuery): RelDeviceType =
    PelagoRelMdDeviceType.table(scan.getTable)

  def deviceType(scan: PelagoTableScan, mq: RelMetadataQuery): RelDeviceType = { //    System.out.println(scan.getDeviceType());
    scan.getDeviceType
  }

  def deviceType(rel: SingleRel, mq: PelagoRelMetadataQuery): RelDeviceType =
    mq.deviceType(rel.getInput)

  def deviceType(rel: Project, mq: RelMetadataQuery): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(rel.getInput)

  def deviceType(rel: BiRel, mq: RelMetadataQuery): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(rel.getRight)

  def deviceType(rel: MultiJoin, mq: RelMetadataQuery): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(rel.getInputs.get(0))

  def deviceType(rel: SetOp, mq: RelMetadataQuery): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(rel.getInputs.get(0))

  def deviceType(values: Values, mq: RelMetadataQuery): RelDeviceType =
    RelDeviceType.ANY

  def deviceType(rel: HepRelVertex, mq: RelMetadataQuery): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(rel.getCurrentRel)

  def deviceType(rel: RelNode, mq: RelMetadataQuery): RelDeviceType = {
    val dtype =
      rel.getTraitSet.getTrait(
        RelDeviceTypeTraitDef.INSTANCE
      ) //TODO: is this safe ? or can it cause an inf loop?
    if (dtype != null) return dtype
    RelDeviceType.X86_64
  }

  def deviceType(
      devcross: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): RelDeviceType =
    devcross.getDeviceType()

  def deviceType(xchange: Exchange, mq: RelMetadataQuery): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(xchange.getInput)

  def deviceType(
      conv: PelagoToEnumerableConverter,
      mq: RelMetadataQuery
  ): RelDeviceType =
    mq.asInstanceOf[PelagoRelMetadataQuery].deviceType(conv)
}
