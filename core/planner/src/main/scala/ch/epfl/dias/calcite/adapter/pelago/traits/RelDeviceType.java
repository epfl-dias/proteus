package ch.epfl.dias.calcite.adapter.pelago.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;

/**
 * Description of the target device of a relational expression.
 */
public class RelDeviceType implements PelagoTrait {
  public static final RelDeviceType X86_64 = new RelDeviceType("X86_64");
  public static final RelDeviceType NVPTX  = new RelDeviceType("NVPTX");
  public static final RelDeviceType ANY    = new RelDeviceType("anydev");

  protected final String dev;

  protected RelDeviceType(String dev) {
    this.dev = dev;
  }

  @Override public String toString() {
    return dev;
  }

  @Override public RelTraitDef getTraitDef() {
    return RelDeviceTypeTraitDef.INSTANCE;
  }

  @Override public boolean satisfies(RelTrait trait) {
    return (this == ANY) || (trait == this);
//    return (trait == this) || (trait == ANY); //(this == ANY) ||
  }

  public Double getMemBW(){
    return ((this == X86_64) ? (100.0 / 10) : 900) * 1024.0 * 1024 * 1024;
  }

  @Override public void register(RelOptPlanner planner) {}
}

// End RelDeviceType.java
