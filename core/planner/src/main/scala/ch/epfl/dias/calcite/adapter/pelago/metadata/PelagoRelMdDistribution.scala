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

import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.{
  BuiltInMetadata,
  MetadataDef,
  MetadataHandler,
  ReflectiveRelMetadataProvider,
  RelMetadataProvider,
  RelMetadataQuery
}
import org.apache.calcite.util.BuiltInMethod

object PelagoRelMdDistribution {
  private val INSTANCE = new PelagoRelMdDistribution
  val SOURCE: RelMetadataProvider =
    ReflectiveRelMetadataProvider.reflectiveSource(
      BuiltInMethod.DISTRIBUTION.method,
      PelagoRelMdDistribution.INSTANCE
    )
}

class PelagoRelMdDistribution
    extends MetadataHandler[BuiltInMetadata.Distribution] {
  override def getDef: MetadataDef[BuiltInMetadata.Distribution] =
    BuiltInMetadata.Distribution.DEF
  def distribution(rel: RelNode, mq: RelMetadataQuery): Null = {
    assert(false)
    null
  }
}
