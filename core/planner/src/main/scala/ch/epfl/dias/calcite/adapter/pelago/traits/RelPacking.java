package ch.epfl.dias.calcite.adapter.pelago.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;

/**
 * Description of the target device of a relational expression.
 */
public class RelPacking implements PelagoTrait {
  public static final RelPacking Packed = new RelPacking("packed");
  public static final RelPacking UnPckd = new RelPacking("unpckd");

  protected final String p;

  protected RelPacking(String dev) {
    this.p = dev;
  }

  @Override public String toString() {
    return p;
  }

  @Override public RelTraitDef getTraitDef() {
    return RelPackingTraitDef.INSTANCE;
  }

  @Override public boolean satisfies(RelTrait trait) {
    return (trait == this);
  }

  @Override public void register(RelOptPlanner planner) {}
}

// End RelPacking.java
