package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.rel.RelNode;

public class RelComputeDeviceTraitDef extends RelTraitDef<RelComputeDevice> {
  public static final RelComputeDeviceTraitDef INSTANCE = new RelComputeDeviceTraitDef();

  protected RelComputeDeviceTraitDef() {}

  @Override public Class<RelComputeDevice> getTraitClass() {
    return RelComputeDevice.class;
  }

  @Override public String getSimpleName() {
    return "compute";
  }

  @Override public RelNode convert(RelOptPlanner planner, RelNode rel, RelComputeDevice toDevice,
                                   boolean allowInfiniteCostConverters) {
    if (rel.getTraitSet().getTrait(INSTANCE).satisfies(toDevice)) return rel;

    //TODO: Can we do something about it? Normally this trait is not convertible, except if we want to nullify it,
    // in which case, we can put a PelagoSplit.
    return null;
  }

  @Override public boolean canConvert(RelOptPlanner planner, RelComputeDevice fromTrait,
      RelComputeDevice toDevice) {
    //See comment in convert(...)
    return fromTrait.satisfies(toDevice);
  }

  @Override public RelComputeDevice getDefault() {
    return RelComputeDevice.X86_64NVPTX;
  }
}

// End RelDeviceTypeTraitDef.java
