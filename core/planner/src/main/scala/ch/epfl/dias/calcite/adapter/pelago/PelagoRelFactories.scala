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

import ch.epfl.dias.calcite.adapter.pelago.rel._
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.Contexts
import org.apache.calcite.rel.{RelCollation, RelNode}
import org.apache.calcite.rel.core.{
  AggregateCall,
  CorrelationId,
  JoinRelType,
  RelFactories
}
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rex.{RexNode, RexUtil}
import org.apache.calcite.tools.RelBuilderFactory
import org.apache.calcite.util.ImmutableBitSet

import java.util

object PelagoRelFactories {
  val PELAGO_PROJECT_FACTORY = new PelagoProjectFactoryImpl
  val PELAGO_FILTER_FACTORY = new PelagoFilterFactoryImpl
  val PELAGO_JOIN_FACTORY = new PelagoJoinFactoryImpl
  val PELAGO_SORT_FACTORY = new PelagoSortFactoryImpl
  val PELAGO_AGGREGATE_FACTORY = new PelagoAggregateFactoryImpl

  val PELAGO_BUILDER: RelBuilderFactory = PelagoRelBuilder.proto(
    Contexts.of(
      PELAGO_PROJECT_FACTORY,
      PELAGO_FILTER_FACTORY,
      PELAGO_JOIN_FACTORY,
      PELAGO_SORT_FACTORY,
      PELAGO_AGGREGATE_FACTORY
    )
  )

  class PelagoProjectFactoryImpl extends RelFactories.ProjectFactory {
    override def createProject(
        input: RelNode,
        hints: util.List[RelHint],
        projects: util.List[_ <: RexNode],
        fieldNames: util.List[String]
    ): PelagoProject = {
      val cluster = input.getCluster
      val rowType = RexUtil.createStructType(
        cluster.getTypeFactory,
        projects,
        fieldNames,
        null
      )
      PelagoProject.create(
        input,
        projects,
        rowType,
        ImmutableList.copyOf[RelHint](hints)
      )
    }
  }

  class PelagoFilterFactoryImpl extends RelFactories.FilterFactory {
    override def createFilter(
        input: RelNode,
        condition: RexNode,
        variablesSet: util.Set[CorrelationId]
    ): PelagoFilter = PelagoFilter.create(input, condition)
  }

  class PelagoJoinFactoryImpl extends RelFactories.JoinFactory {
    override def createJoin(
        left: RelNode,
        right: RelNode,
        hints: util.List[RelHint],
        condition: RexNode,
        variablesSet: util.Set[CorrelationId],
        joinType: JoinRelType,
        semiJoinDone: Boolean
    ): PelagoJoin =
      PelagoJoin.create(left, right, condition, variablesSet, joinType)
  }

  class PelagoSortFactoryImpl extends RelFactories.SortFactory {
    override def createSort(
        input: RelNode,
        collation: RelCollation,
        offset: RexNode,
        fetch: RexNode
    ): PelagoSort = PelagoSort.create(input, collation, offset, fetch)
  }

  class PelagoAggregateFactoryImpl extends RelFactories.AggregateFactory {
    override def createAggregate(
        input: RelNode,
        hints: util.List[RelHint],
        groupSet: ImmutableBitSet,
        groupSets: ImmutableList[ImmutableBitSet],
        aggCalls: util.List[AggregateCall]
    ): PelagoAggregate =
      PelagoAggregate.create(
        input,
        ImmutableList.copyOf[RelHint](hints),
        groupSet,
        groupSets,
        aggCalls
      )
  }
}
