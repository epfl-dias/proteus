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

package ch.epfl.dias.calcite.adapter.pelago

import org.apache.calcite.plan.Context
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptSchema
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.server.CalciteServerStatement
import org.apache.calcite.tools.FrameworkConfig
import org.apache.calcite.tools.Frameworks

import org.apache.calcite.tools.RelBuilder
import org.apache.calcite.tools.RelBuilderFactory

object PelagoRelBuilder {
  def create(config: FrameworkConfig): PelagoRelBuilder = {
    val clusters = Array[RelOptCluster](null)
    val relOptSchemas = Array[RelOptSchema](null)

    Frameworks.withPrepare(new Frameworks.BasePrepareAction[Void]() {
      override def apply(
          cluster: RelOptCluster,
          relOptSchema: RelOptSchema,
          rootSchema: SchemaPlus,
          statement: CalciteServerStatement
      ): Null = {
        clusters(0) = cluster
        relOptSchemas(0) = relOptSchema
        null
      }
    })
    new PelagoRelBuilder(config.getContext, clusters(0), relOptSchemas(0))
  }

  def proto(context: Context): RelBuilderFactory =
    new RelBuilderFactory() {
      override def create(
          cluster: RelOptCluster,
          schema: RelOptSchema
      ): RelBuilder = new PelagoRelBuilder(context, cluster, schema)
    }
}

class PelagoRelBuilder private (
    context: Context,
    cluster: RelOptCluster,
    relOptSchema: RelOptSchema
) extends RelBuilder(context, cluster, relOptSchema) {}
