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

package ch.epfl.dias.calcite.adapter.pelago.traits

import org.apache.calcite.plan.{RelOptPlanner, RelTrait}
import org.apache.calcite.rel.{RelDistribution, RelDistributions}

import java.util

/**
  * Description of the distribution across homogeneous devices, of an input stream.
  */
object RelHomDistribution {
  protected val available_distributions =
    new util.HashMap[RelDistribution, RelHomDistribution]
  val RANDOM =
    new RelHomDistribution("homRandom", RelDistributions.RANDOM_DISTRIBUTED)
  val BRDCST =
    new RelHomDistribution("homBrdcst", RelDistributions.BROADCAST_DISTRIBUTED)
  val SINGLE = new RelHomDistribution("homSingle", RelDistributions.SINGLETON)
  def from(distr: RelDistribution): RelHomDistribution =
    available_distributions.get(distr)
}
class RelHomDistribution protected (
    val str: String,
    val distribution: RelDistribution
) extends PelagoTrait { //Check that we do not already have a distribution with the same RelDistribution
  assert(
    !RelHomDistribution.available_distributions.containsKey(distribution)
  )
  RelHomDistribution.available_distributions.put(distribution, this)
  override def toString: String = str
  override def getTraitDef: RelHomDistributionTraitDef =
    RelHomDistributionTraitDef.INSTANCE
  def getDistribution: RelDistribution = distribution
  override def satisfies(`trait`: RelTrait): Boolean = `trait` eq this
  override def register(planner: RelOptPlanner): Unit = {}
}
