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
import org.apache.calcite.rel.core.{Aggregate, Join}
import org.apache.calcite.rel.metadata._
import org.apache.calcite.util.BuiltInMethod

import java.util.Objects

object PelagoRelMdRowCount {
  private val INSTANCE = new PelagoRelMdRowCount
  val SOURCE: RelMetadataProvider = ChainedRelMetadataProvider.of(
    ImmutableList.of(
      ReflectiveRelMetadataProvider.reflectiveSource(
        BuiltInMethod.ROW_COUNT.method,
        PelagoRelMdRowCount.INSTANCE
      ),
      RelMdRowCount.SOURCE
    )
  )
}
class PelagoRelMdRowCount protected ()
    extends MetadataHandler[BuiltInMetadata.RowCount] {
  final private val `def` = new RelMdRowCount

  override def getDef: MetadataDef[BuiltInMetadata.RowCount] =
    BuiltInMetadata.RowCount.DEF

  def getRowCount(rel: PelagoUnnest, mq: RelMetadataQuery): java.lang.Double =
    rel.estimateRowCount(mq)

  def getRowCount(rel: PelagoPack, mq: RelMetadataQuery): java.lang.Double = { //    return Math.ceil(mq.getRowCount(rel.getInput()) / (1024*1024.0));
    mq.getRowCount(rel.getInput) / CostModel.blockSize
  }

  def getRowCount(rel: Aggregate, mq: RelMetadataQuery): java.lang.Double = {
    if (rel.getGroupCount == 0) return 1.0
    val groupKey = rel.getGroupSet
// rowCount is the cardinality of the group by columns
    val distinctRowCount = mq.getDistinctRowCount(rel.getInput, groupKey, null)
// groupby's are generally very selective
    Objects.requireNonNullElseGet(
      distinctRowCount,
      () => `def`.getRowCount(rel, mq) / 10
    )
  }

  def getRowCount(rel: PelagoUnpack, mq: RelMetadataQuery): java.lang.Double =
    mq.getRowCount(rel.getInput) * CostModel.blockSize

  def getRowCount(rel: PelagoRouter, mq: RelMetadataQuery): java.lang.Double =
    rel.estimateRowCount(mq) //mq.getRowCount(rel.getInput()) / 2;

  def getRowCount(rel: PelagoUnion, mq: RelMetadataQuery): java.lang.Double =
    mq.getRowCount(rel.getInput(0)) + mq.getRowCount(rel.getInput(1))

  def getRowCount(
      rel: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): java.lang.Double =
    mq.getRowCount(rel.getInput)

  def getRowCount(rel: PelagoSplit, mq: RelMetadataQuery): java.lang.Double =
    mq.getRowCount(rel.getInput(0)) / 2

  def getRowCount(
      rel: PelagoTableScan,
      mq: RelMetadataQuery
  ): java.lang.Double = {
    var rc = rel.getTable.getRowCount
// FIXME: using CostModel.blockSize() breaks jo/in ordering in ssbm100
    if (rel.getTraitSet.containsIfApplicable(RelPacking.Packed))
      rc /= CostModel.blockSize
    rc
  }

  def getRowCount(rel: Join, mq: RelMetadataQuery): java.lang.Double = {
    val leftRows = mq.getRowCount(rel.getLeft)
    val rightRows = mq.getRowCount(rel.getRight)
    val rows = Math.max(leftRows, rightRows)
    val small =
      if (leftRows < rightRows) rel.getLeft
      else rel.getRight
    val sel = mq.getPercentageOriginalRows(small)
    if (sel == null) return rows
    rows * sel
  }
}
