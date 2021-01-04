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

import ch.epfl.dias.calcite.adapter.pelago.rel._
import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelComputeDevice,
  RelDeviceType,
  RelPacking
}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.{
  MetadataDef,
  MetadataHandler,
  ReflectiveRelMetadataProvider,
  RelMetadataProvider,
  RelMetadataQuery
}

import scala.collection.JavaConverters._

object PelagoRelMdComputeDevice {
  private val INSTANCE = new PelagoRelMdComputeDevice
  val SOURCE: RelMetadataProvider =
    ReflectiveRelMetadataProvider.reflectiveSource(
      ComputeDevice.method,
      PelagoRelMdComputeDevice.INSTANCE
    )
}
class PelagoRelMdComputeDevice extends MetadataHandler[ComputeDevice] {
  override def getDef: MetadataDef[ComputeDevice] = ComputeDevice.DEF

  def computeType(rel: RelNode, mq: RelMetadataQuery): RelComputeDevice = {
    if (rel.getTraitSet.containsIfApplicable(RelDeviceType.X86_64))
      return RelComputeDevice.X86_64
    else if (rel.getTraitSet.containsIfApplicable(RelDeviceType.NVPTX))
      return RelComputeDevice.NVPTX
    RelComputeDevice.from(
      rel.getInputs.asScala
        .map((e: RelNode) =>
          mq.asInstanceOf[PelagoRelMetadataQuery].computeType(e)
        )
        .toList
    )
  }

  def computeType(
      scan: PelagoTableScan,
      mq: RelMetadataQuery
  ): RelComputeDevice = { //if the scan is producing tuples, then it actually runs on the device its DeviceTypeTrait specifies
    if (
      mq.asInstanceOf[PelagoRelMetadataQuery]
        .packing(scan)
        .satisfies(RelPacking.UnPckd)
    )
      if (scan.getDeviceType.satisfies(RelDeviceType.X86_64))
        return RelComputeDevice.X86_64
      else if (scan.getDeviceType.satisfies(RelDeviceType.NVPTX))
        return RelComputeDevice.NVPTX
    RelComputeDevice.NONE
  }

  def computeType(split: PelagoSplit, mq: RelMetadataQuery): RelComputeDevice =
    RelComputeDevice.NONE

  def computeType(split: PelagoUnion, mq: RelMetadataQuery): RelComputeDevice =
    RelComputeDevice.NONE

  def computeType(rel: PelagoRouter, mq: RelMetadataQuery): RelComputeDevice =
    mq.asInstanceOf[PelagoRelMetadataQuery].computeType(rel.getInput)

  def computeType(
      rel: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): RelComputeDevice =
    mq.asInstanceOf[PelagoRelMetadataQuery].computeType(rel.getInput)
}
