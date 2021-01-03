package ch.epfl.dias.calcite.adapter.pelago.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableList;

import java.util.List;

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
//    if (rel.getTraitSet().getTrait(INSTANCE).satisfies(toDevice)) return rel;

    List<RelNode> inputs = rel.getInputs();
    if (inputs.isEmpty()) return null;

    RelDeviceType dev = (toDevice == RelComputeDevice.NVPTX) ? RelDeviceType.NVPTX : RelDeviceType.X86_64;
    ImmutableList.Builder<RelNode> b = ImmutableList.builder();
    for (RelNode inp: inputs){
      b.add(planner.changeTraits(inp, inp.getTraitSet().replace(dev)));
    }

    RelNode newRel = rel.copy(null, b.build());
    newRel = planner.register(newRel, rel);
    if (!newRel.getTraitSet().contains(toDevice)) return null;
    RelTraitSet traitSet = rel.getTraitSet().replace(toDevice);
    if (!newRel.getTraitSet().equals(traitSet)) {
      newRel = planner.changeTraits(newRel, traitSet);
    }
    return newRel;
  }

  @Override public boolean canConvert(RelOptPlanner planner, RelComputeDevice fromTrait,
      RelComputeDevice toDevice) {
    //See comment in convert(...)
    return false;//toDevice != RelComputeDevice.X86_64NVPTX && toDevice != RelComputeDevice.NONE;//fromTrait.satisfies(toDevice);
  }

  @Override public RelComputeDevice getDefault() {
    return RelComputeDevice.X86_64NVPTX;
  }
}

// End RelDeviceTypeTraitDef.java
