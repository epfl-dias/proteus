package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistributions;
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

  @Override public RelNode convert(RelOptPlanner planner, RelNode rel, RelHetDistribution distribution,
                                   boolean allowInfiniteCostConverters) {
    if (!rel.getTraitSet().containsIfApplicable(RelDistributions.SINGLETON) ||
        !rel.getTraitSet().containsIfApplicable(RelDeviceType.X86_64) ||
        rel.getConvention() != PelagoRel.CONVENTION){
      return null;
    }

    final PelagoSplit split = PelagoSplit.create(rel, distribution);

    RelNode newRel = planner.register(split, rel);
    RelTraitSet traitSet = rel.getTraitSet().replace(distribution);
    if (!newRel.getTraitSet().equals(traitSet)) {
      newRel = planner.changeTraits(newRel, traitSet);
    }
    return newRel;

  }

  @Override public boolean canConvert(RelOptPlanner planner, RelHetDistribution fromTrait,
      RelHetDistribution toTrait) {
    return toTrait != RelHetDistribution.SINGLETON;
  }

  @Override public RelHetDistribution getDefault() {
    return RelHetDistribution.SINGLETON;
  }
}

// End RelDeviceTypeTraitDef.java
