package org.apache.calcite.plan.volcano;

import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptCostFactory;

public class PelagoCostFactory implements RelOptCostFactory {
  public static final PelagoCostFactory INSTANCE = new PelagoCostFactory();

  private PelagoCostFactory(){}

  @Override public RelOptCost makeCost(final double rowCount, final double cpu, final double io) {
//    return new PelagoCost(rowCount + 1e-3 * cpu + 1e3 * io, cpu, io);
    return new PelagoCost(io, cpu, rowCount);
//    return new PelagoCost(rowCount, cpu, io);
  }

  @Override public RelOptCost makeHugeCost() {
    return PelagoCost.HUGE;
  }

  @Override public RelOptCost makeInfiniteCost() {
    return PelagoCost.INFINITY;
  }

  @Override public RelOptCost makeTinyCost() {
    return PelagoCost.TINY;
  }

  @Override public RelOptCost makeZeroCost() {
    return PelagoCost.ZERO;
  }
}
