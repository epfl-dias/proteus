package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.core.DeviceCross;
//import org.apache.calcite.rel.logical.LogicalDeviceCross;

/**
 * Definition of the device type trait.
 *
 * <p>Target device type is a physical property (i.e. a trait) because it can be
 * changed without loss of information. The converter to do this is the
 * {@link PelagoDeviceCross} operator.
 */
public class RelDeviceTypeTraitDef extends RelTraitDef<RelDeviceType> {
  public static final RelDeviceTypeTraitDef INSTANCE = new RelDeviceTypeTraitDef();

  protected RelDeviceTypeTraitDef() {}

  @Override public Class<RelDeviceType> getTraitClass() {
    return RelDeviceType.class;
  }

  @Override public String getSimpleName() {
    return "device";
  }

  @Override public RelNode convert(RelOptPlanner planner, RelNode rel, RelDeviceType toDevice,
                                   boolean allowInfiniteCostConverters) {
    if (toDevice == RelDeviceType.ANY || rel.getTraitSet().contains(toDevice)) {
      return rel;
    }

    final PelagoDeviceCross crossDev = PelagoDeviceCross.create(rel, toDevice);
    RelNode newRel = planner.register(crossDev, rel);
    final RelTraitSet newTraitSet = rel.getTraitSet().replace(toDevice);
    if (!newRel.getTraitSet().equals(newTraitSet)) {
      newRel = planner.changeTraits(newRel, newTraitSet);
    }
    return newRel;


//    return PelagoDeviceCross.create(planner.changeTraits(rel, PelagoRel.CONVENTION()), toDevice);
  }

  @Override public boolean canConvert(RelOptPlanner planner, RelDeviceType fromTrait,
                                      RelDeviceType toDevice) {
    return fromTrait != toDevice;
  }

  @Override public RelDeviceType getDefault() {
    return RelDeviceType.X86_64;
  }
}

// End RelDeviceTypeTraitDef.java
