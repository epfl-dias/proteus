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

package org.apache.calcite.plan

import ch.epfl.dias.calcite.adapter.pelago.metadata.{
  PelagoRelMetadataProvider,
  PelagoRelMetadataQuery
}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.`type`.RelDataTypeFactory
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider
import org.apache.calcite.rex.RexBuilder

import java.util
import java.util.concurrent.atomic.AtomicInteger

object PelagoRelOptCluster {
  def create(planner: RelOptPlanner, rexBuilder: RexBuilder) =
    new PelagoRelOptCluster(
      planner,
      rexBuilder.getTypeFactory,
      rexBuilder,
      new AtomicInteger(0),
      new util.HashMap[String, RelNode]
    )
}

class PelagoRelOptCluster private[plan] (
    val planner: RelOptPlanner,
    val typeFactory: RelDataTypeFactory,
    val rexBuilder: RexBuilder,
    val nextCorrel: AtomicInteger,
    val mapCorrelToRel: util.Map[String, RelNode]
) extends RelOptCluster(
      planner,
      typeFactory,
      rexBuilder,
      nextCorrel,
      mapCorrelToRel
    ) {
  super.setMetadataProvider(PelagoRelMetadataProvider.INSTANCE)
  setMetadataQuerySupplier(() => {
    super.setMetadataProvider(
      ChainedRelMetadataProvider
        .of(
          util.List.of(
            getMetadataProvider,
            PelagoRelMetadataProvider.INSTANCE
          )
        )
    )
    PelagoRelMetadataQuery.instance
  })
}
