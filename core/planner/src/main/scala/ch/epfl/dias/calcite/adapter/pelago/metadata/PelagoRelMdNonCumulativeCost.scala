/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *//*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

class PelagoRelMdNonCumulativeCost protected() extends MetadataHandler[BuiltInMetadata.NonCumulativeCost] {
  override def getDef: MetadataDef[BuiltInMetadata.NonCumulativeCost] = BuiltInMetadata.NonCumulativeCost.DEF

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

  protected def computeAndCacheNonCumulativeCost[T <: RelNode](rel: T, mq: RelMetadataQuery): RelOptCost = {
//    if (rel.isInstanceOf[PelagoUnion])
      cache.remove(rel)
    mq.clearCache(rel)
    try {
      cache.getOrElseUpdate(rel, {
        val rowCount = mq.getRowCount(rel)
        assert(rowCount != null)
        val c = CostModel.getNonCumulativeCost(rel)
        if (c == null) {
          assert(!rel.isInstanceOf[PelagoRel])
          rel.computeSelfCost(rel.getCluster.getPlanner, mq).multiplyBy(1e200)
        } else {
          c.toRelCost(rel.getCluster.getPlanner.getCostFactory, rel.getTraitSet, rowCount)
        }
      })
    } catch {
      case _: CyclicMetadataException => rel.getCluster.getPlanner.getCostFactory.makeInfiniteCost()
    }
  }


  def getNonCumulativeCost(rel: PelagoRel, mq: RelMetadataQuery): RelOptCost = {
    computeAndCacheNonCumulativeCost(rel, mq)
  }
}

// End PelagoRelMdSelfCost.scala