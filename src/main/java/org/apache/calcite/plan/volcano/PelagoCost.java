package org.apache.calcite.plan.volcano;

import org.apache.calcite.plan.RelOptCost;

public class PelagoCost extends VolcanoCost {
  static final PelagoCost INFINITY =
      new PelagoCost(
          Double.POSITIVE_INFINITY,
          Double.POSITIVE_INFINITY,
          Double.POSITIVE_INFINITY) {
        public String toString() {
          return "{pinf}";
        }
      };

  static final PelagoCost HUGE =
      new PelagoCost(Double.MAX_VALUE, Double.MAX_VALUE, Double.MAX_VALUE) {
        public String toString() {
          return "{phuge}";
        }
      };

  static final PelagoCost ZERO =
      new PelagoCost(0.0, 0.0, 0.0) {
        public String toString() {
          return "{p0}";
        }
      };

  static final PelagoCost TINY =
      new PelagoCost(1.0, 1.0, 0.0) {
        public String toString() {
          return "{ptiny}";
        }
      };

  public PelagoCost(final double rowCount, final double cpu, final double io) {
    super(rowCount, cpu, io);
  }
}
