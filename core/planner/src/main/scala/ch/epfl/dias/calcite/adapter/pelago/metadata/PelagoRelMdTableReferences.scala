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
import org.apache.calcite.rex.RexTableInputRef
import org.apache.calcite.util.BuiltInMethod

import java.util

object PelagoRelMdTableReferences {
  private val INSTANCE = new PelagoRelMdTableReferences
  val SOURCE: RelMetadataProvider = ChainedRelMetadataProvider.of(
    ImmutableList.of(
      ReflectiveRelMetadataProvider.reflectiveSource(
        BuiltInMethod.TABLE_REFERENCES.method,
        PelagoRelMdTableReferences.INSTANCE
      ),
      RelMdTableReferences.SOURCE
    )
  )
}

class PelagoRelMdTableReferences protected ()
    extends MetadataHandler[BuiltInMetadata.TableReferences] {

  override def getDef: MetadataDef[BuiltInMetadata.TableReferences] =
    BuiltInMetadata.TableReferences.DEF

  def getTableReferences(
      rel: PelagoUnpack,
      mq: RelMetadataQuery
  ): util.Set[RexTableInputRef.RelTableRef] =
    mq.getTableReferences(rel.getInput)

  def getTableReferences(
      rel: PelagoPack,
      mq: RelMetadataQuery
  ): util.Set[RexTableInputRef.RelTableRef] =
    mq.getTableReferences(rel.getInput)

  def getTableReferences(
      rel: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): util.Set[RexTableInputRef.RelTableRef] =
    mq.getTableReferences(rel.getInput)

  def getTableReferences(
      rel: PelagoRouter,
      mq: RelMetadataQuery
  ): util.Set[RexTableInputRef.RelTableRef] =
    mq.getTableReferences(rel.getInput)
}
