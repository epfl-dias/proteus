package ch.epfl.dias.calcite.adapter.pelago.costs

import ch.epfl.dias.calcite.adapter.pelago.traits.{RelDeviceTypeTraitDef, RelHetDistribution}
import org.apache.calcite.plan.{RelOptCost, RelOptCostFactory, RelTraitSet}

sealed trait Cost {
  def toRelCost(factory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost

  def +(other: Cost): Cost = {
    CombinedCost(this, other)
  }
}

case class MemBW(bytesPerTuple: Double) extends Cost {
  override def toRelCost(costFactory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost = costFactory.makeCost(
    rowCount,
    1,
    (bytesPerTuple * rowCount) / (traitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE).getMemBW * {if (!traitSet.containsIfApplicable(RelHetDistribution.SINGLETON)) 1.5 else 1})
  )
}

case class RandomMemBW(bytesPerTuple: Double) extends Cost {
  override def toRelCost(factory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost = {
    MemBW(bytesPerTuple * 8).toRelCost(factory, traitSet, rowCount)
  }
}

case class InterconnectBW(bytesPerTuple: Double) extends Cost {
  override def toRelCost(costFactory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost = costFactory.makeCost(
    rowCount,
    1,
    (bytesPerTuple * rowCount) / (16 * 1024.0 * 1024.0 * 1024.0 * {if (!traitSet.containsIfApplicable(RelHetDistribution.SINGLETON)) 1.5 else 1})
  )
}

case class Compute(cyclesPerTuple: Double) extends Cost {
  override def toRelCost(costFactory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost = costFactory.makeCost(
    rowCount,
    cyclesPerTuple * rowCount / CostModel.getDevCount(traitSet),
    0
  )
}

case class CombinedCost(cost0: Cost, cost1: Cost) extends Cost {
  override def toRelCost(factory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost = {
    cost0.toRelCost(factory, traitSet, rowCount).plus(cost1.toRelCost(factory, traitSet, rowCount))
  }
}

case class InfiniteCost() extends Cost {
  override def toRelCost(factory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost = factory.makeHugeCost()
}


case class ReallyInfiniteCost() extends Cost {
  override def toRelCost(factory: RelOptCostFactory, traitSet: RelTraitSet, rowCount: Double): RelOptCost = factory.makeInfiniteCost()
}