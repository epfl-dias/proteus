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
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoRel
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata._
import org.apache.calcite.util.BuiltInMethod

import scala.collection.mutable

/**
  * RelNodes supply a function {@link RelNode#computeSelfCost(RelOptPlanner, RelMetadataQuery)} to compute selfCost
  */
object PelagoRelMdNonCumulativeCost {
  val SOURCE: RelMetadataProvider =
    ReflectiveRelMetadataProvider.reflectiveSource(
      BuiltInMethod.NON_CUMULATIVE_COST.method,
      new PelagoRelMdNonCumulativeCost
    )
}

class PelagoRelMdNonCumulativeCost protected ()
    extends MetadataHandler[BuiltInMetadata.NonCumulativeCost] {
  override def getDef: MetadataDef[BuiltInMetadata.NonCumulativeCost] =
    BuiltInMetadata.NonCumulativeCost.DEF

  private val cache = mutable.WeakHashMap[RelNode, RelOptCost]()

  /** Fallback method to deduce selfCost for any relational expression not
    * handled by a more specific method.
    *
    * @param rel Relational expression
    * @return Relational expression's self cost
    */
  def getNonCumulativeCost(rel: RelNode, mq: RelMetadataQuery): RelOptCost = {
    computeAndCacheNonCumulativeCost(rel, mq)
  }

  protected def computeAndCacheNonCumulativeCost[T <: RelNode](
      rel: T,
      mq: RelMetadataQuery
  ): RelOptCost = {
//    if (rel.isInstanceOf[PelagoUnion])
    cache.remove(rel)
    mq.clearCache(rel)
    try {
      cache.getOrElseUpdate(
        rel, {
          val rowCount = mq.getRowCount(rel)
          assert(rowCount != null)
          val c = CostModel.getNonCumulativeCost(rel)
          if (c == null) {
            assert(!rel.isInstanceOf[PelagoRel])
            rel.computeSelfCost(
              rel.getCluster.getPlanner,
              mq
            ) //.multiplyBy(1e200)
          } else {
            c.toRelCost(
              rel.getCluster.getPlanner.getCostFactory,
              rel.getTraitSet,
              rowCount
            )
          }
        }
      )
    } catch {
      case _: CyclicMetadataException =>
        rel.getCluster.getPlanner.getCostFactory.makeInfiniteCost()
    }
  }

  def getNonCumulativeCost(rel: PelagoRel, mq: RelMetadataQuery): RelOptCost = {
    computeAndCacheNonCumulativeCost(rel, mq)
  }
}

// End PelagoRelMdSelfCost.scala
