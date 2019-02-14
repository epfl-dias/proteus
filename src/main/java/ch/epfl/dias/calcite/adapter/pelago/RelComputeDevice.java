package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;

import com.google.common.collect.ImmutableList;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * TODO: should we convert it into a RelMultipleTrait ? Does a RelMultipleTrait has *ANY* of the values or all?
 */
public class RelComputeDevice implements PelagoTrait {
  public static final RelComputeDevice NONE        = new RelComputeDevice("none");
  public static final RelComputeDevice X86_64      = new RelComputeDevice("cX86_64");
  public static final RelComputeDevice NVPTX       = new RelComputeDevice("cNVPTX");
  public static final RelComputeDevice X86_64NVPTX = new RelComputeDevice("cNV+X86");

  protected final String computeTypes;

  protected RelComputeDevice(String computeTypes) {
    this.computeTypes = computeTypes;
  }

  public static RelComputeDevice from(RelDeviceType dev){
    if (dev == RelDeviceType.X86_64){
      return RelComputeDevice.X86_64;
    } else if (dev == RelDeviceType.NVPTX){
      return RelComputeDevice.NVPTX;
    }
    assert(false);
    return RelComputeDevice.X86_64NVPTX;
  }

  public static RelComputeDevice from(RelNode input) {
    RelTraitSet trait = input.getTraitSet();
    RelDeviceType dev = trait.getTrait(RelDeviceTypeTraitDef.INSTANCE);
    RelComputeDevice comp = RelComputeDevice.from(input, false);
    return RelComputeDevice.from(ImmutableList.of(RelComputeDevice.from(dev), comp).stream());
  }

  public static RelComputeDevice from(RelNode input, boolean isCompute) {
    if (isCompute) return from(input);
    RelTraitSet trait = input.getTraitSet();
    return trait.getTrait(RelComputeDeviceTraitDef.INSTANCE);
  }

  public static RelComputeDevice from(final Stream<RelComputeDevice> relComputeDeviceStream) {
    List<RelComputeDevice> devs = relComputeDeviceStream.distinct().filter((e) -> e != NONE).collect(Collectors.toList());
    if (devs.size() == 0) return NONE;
    if (devs.size() == 1) return devs.get(0);
    return X86_64NVPTX; //NOTE: do not forget to update this if you add devices!
  }

  @Override public String toString() {
    return computeTypes;
  }

  @Override public RelTraitDef getTraitDef() {
    return RelComputeDeviceTraitDef.INSTANCE;
  }

  @Override public boolean satisfies(RelTrait trait) {
    if (trait == this) return true;  // everything satisfies itself... (Singleton)
    if (this  == NONE) return true;  // no processing can be considered as any processing
    if (trait == NONE) return false; // only itself satisfies NONE
    if (trait == X86_64NVPTX) {
      // for now, returned expression is always true, but leave it here for future-proofness.
      return (this == X86_64 || this == NVPTX);
    }
    return false;
  }

  @Override public void register(RelOptPlanner planner) {}
}
