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

/**
  * Description of the distribution across device types of an input stream.
  */
object RelHetDistribution {
  val SPLIT = new RelHetDistribution("hetSplit")
  val SPLIT_BRDCST = new RelHetDistribution("hetBrdcst")
  val SINGLETON = new RelHetDistribution("hetSingle")
}

class RelHetDistribution protected (val distr: String) extends PelagoTrait {
  def getNumOfDeviceTypes: Int =
    if (this eq RelHetDistribution.SINGLETON) 1
    else 2

  override def toString: String = distr

  override def getTraitDef: RelHetDistributionTraitDef =
    RelHetDistributionTraitDef.INSTANCE

  override def satisfies(`trait`: RelTrait): Boolean = `trait` eq this

  override def register(planner: RelOptPlanner): Unit = {}
}
