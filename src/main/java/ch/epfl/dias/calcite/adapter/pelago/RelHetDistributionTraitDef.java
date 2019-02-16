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
    if (rel.getConvention() != PelagoRel.CONVENTION){
      return null;
    }

    RelTraitSet inptraitSet = rel.getTraitSet().replace(RelDeviceType.X86_64).replace(RelHomDistribution.SINGLE);
    RelTraitSet traitSet = rel.getTraitSet().replace(distribution);
    RelNode input = rel;
    if (!rel.getTraitSet().equals(inptraitSet)) {
      input = planner.register(planner.changeTraits(rel, inptraitSet), rel);
    }

    final PelagoSplit router = PelagoSplit.create(input, distribution);

    RelNode newRel = planner.register(router, rel);
    if (!newRel.getTraitSet().equals(traitSet)) {
      newRel = planner.register(planner.changeTraits(newRel, traitSet), rel);
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
