package org.apache.calcite.plan.volcano;

import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptUtil;

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
      new PelagoCost(1, 1.0, 0.0) {
        public String toString() {
          return "{ptiny}";
        }
      };

  public PelagoCost(final double rowCount, final double cpu, final double io) {
    super(rowCount, cpu, io);
  }

  public String toString() {
    return "{" + rowCount + "s (io), " + cpu + " cpu, " + io + " rows}";
  }

  public RelOptCost plus(RelOptCost other) {
    PelagoCost that = (PelagoCost) other;
    if ((this == INFINITY) || (that == INFINITY)) {
      return INFINITY;
    }
    return new PelagoCost(
        this.rowCount + that.rowCount,
        this.cpu + that.cpu,
        this.io + that.io);
  }

  public boolean isEqWithEpsilon(RelOptCost other) {
    if (!(other instanceof PelagoCost)) {
      return false;
    }
    PelagoCost that = (PelagoCost) other;
    return (this == that)
        || ((Math.abs(this.rowCount - that.rowCount) < RelOptUtil.EPSILON)
        && (Math.abs(this.cpu - that.cpu) < RelOptUtil.EPSILON)
        && (Math.abs(this.io - that.io) < RelOptUtil.EPSILON));
  }
}
