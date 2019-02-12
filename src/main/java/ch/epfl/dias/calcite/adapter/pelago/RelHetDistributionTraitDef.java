package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.rel.RelNode;

public class RelHetDistributionTraitDef extends RelTraitDef<RelHetDistribution> {
  public static final RelHetDistributionTraitDef INSTANCE = new RelHetDistributionTraitDef();

  protected RelHetDistributionTraitDef() {}

  @Override public Class<RelHetDistribution> getTraitClass() {
    return RelHetDistribution.class;
  }

  @Override public String getSimpleName() {
    return "het_distribution";
  }

  @Override public RelNode convert(RelOptPlanner planner, RelNode rel, RelHetDistribution toDevice,
                                   boolean allowInfiniteCostConverters) {
    return null;
  }

  @Override public boolean canConvert(RelOptPlanner planner, RelHetDistribution fromTrait,
      RelHetDistribution toDevice) {
    return false;
  }

  @Override public RelHetDistribution getDefault() {
    return RelHetDistribution.SINGLETON;
  }
}

// End RelDeviceTypeTraitDef.java
