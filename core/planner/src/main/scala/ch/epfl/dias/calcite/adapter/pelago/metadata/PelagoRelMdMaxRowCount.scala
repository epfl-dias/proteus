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

import ch.epfl.dias.calcite.adapter.pelago.costs.CostModel
import ch.epfl.dias.calcite.adapter.pelago.rel._
import ch.epfl.dias.calcite.adapter.pelago.traits.RelPacking
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.metadata._
import org.apache.calcite.util.BuiltInMethod

import java.lang

object PelagoRelMdMaxRowCount {
  private val INSTANCE = new PelagoRelMdMaxRowCount
  val SOURCE: RelMetadataProvider = ChainedRelMetadataProvider.of(
    ImmutableList.of(
      ReflectiveRelMetadataProvider.reflectiveSource(
        BuiltInMethod.MAX_ROW_COUNT.method,
        PelagoRelMdMaxRowCount.INSTANCE
      ),
      RelMdMaxRowCount.SOURCE
    )
  )
}

class PelagoRelMdMaxRowCount protected ()
    extends MetadataHandler[BuiltInMetadata.MaxRowCount] {
  override def getDef: MetadataDef[BuiltInMetadata.MaxRowCount] =
    BuiltInMetadata.MaxRowCount.DEF

  def getMaxRowCount(rel: PelagoPack, mq: RelMetadataQuery): lang.Double =
    Math.ceil(mq.getMaxRowCount(rel.getInput) / CostModel.blockSize)

  def getMaxRowCount(rel: PelagoUnpack, mq: RelMetadataQuery): lang.Double =
    mq.getMaxRowCount(rel.getInput) * CostModel.blockSize

  def getMaxRowCount(rel: PelagoUnnest, mq: RelMetadataQuery): lang.Double =
    java.lang.Double.POSITIVE_INFINITY

  def getMaxRowCount(rel: PelagoRouter, mq: RelMetadataQuery): lang.Double =
    mq.getMaxRowCount(rel.getInput) //mq.getRowCount(rel.getInput()) / 2;

  def getMaxRowCount(rel: PelagoUnion, mq: RelMetadataQuery): lang.Double =
    mq.getMaxRowCount(rel.getInput(0)) + mq.getMaxRowCount(rel.getInput(1))

  def getMaxRowCount(
      rel: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): lang.Double =
    mq.getMaxRowCount(rel.getInput)

  def getMaxRowCount(
      rel: PelagoTableScan,
      mq: RelMetadataQuery
  ): lang.Double = {
    var rc = rel.getTable.getRowCount
    if (rel.getTraitSet.containsIfApplicable(RelPacking.Packed))
      rc /= CostModel.blockSize
    rc
  }
}
