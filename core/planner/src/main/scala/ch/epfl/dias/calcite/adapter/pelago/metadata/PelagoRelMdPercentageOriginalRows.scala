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
  PelagoRouter,
  PelagoUnpack
}
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.metadata._
import org.apache.calcite.util.BuiltInMethod

import java.lang

object PelagoRelMdPercentageOriginalRows {
  private val INSTANCE = new PelagoRelMdPercentageOriginalRows
  val SOURCE: RelMetadataProvider = ChainedRelMetadataProvider.of(
    ImmutableList.of(
      ReflectiveRelMetadataProvider.reflectiveSource(
        BuiltInMethod.PERCENTAGE_ORIGINAL_ROWS.method,
        PelagoRelMdPercentageOriginalRows.INSTANCE
      ),
      RelMdPercentageOriginalRows.SOURCE
    )
  )
}

class PelagoRelMdPercentageOriginalRows private ()
    extends MetadataHandler[BuiltInMetadata.PercentageOriginalRows] {
//~ Methods ----------------------------------------------------------------
  override def getDef: MetadataDef[BuiltInMetadata.PercentageOriginalRows] =
    BuiltInMetadata.PercentageOriginalRows.DEF

  def getPercentageOriginalRows(
      rel: PelagoUnpack,
      mq: RelMetadataQuery
  ): lang.Double =
    mq.getPercentageOriginalRows(rel.getInput)

  def getPercentageOriginalRows(
      rel: PelagoPack,
      mq: RelMetadataQuery
  ): lang.Double =
    mq.getPercentageOriginalRows(rel.getInput)

  def getPercentageOriginalRows(
      rel: PelagoRouter,
      mq: RelMetadataQuery
  ): lang.Double =
    mq.getPercentageOriginalRows(rel.getInput)

  def getPercentageOriginalRows(
      rel: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): lang.Double =
    mq.getPercentageOriginalRows(rel.getInput)
}
