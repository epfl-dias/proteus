package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;

public class RelHomDistributionTraitDef extends RelTraitDef<RelHomDistribution> {
  public static final RelHomDistributionTraitDef INSTANCE = new RelHomDistributionTraitDef();

  protected RelHomDistributionTraitDef() {}

  @Override public Class<RelHomDistribution> getTraitClass() {
    return RelHomDistribution.class;
  }

  @Override public String getSimpleName() {
    return "hom_distribution";
  }

  @Override public RelNode convert(RelOptPlanner planner, RelNode rel, RelHomDistribution distribution,
                                   boolean allowInfiniteCostConverters) {
    if (rel.getTraitSet().containsIfApplicable(distribution)) return rel;

    if (!rel.getTraitSet().containsIfApplicable(RelDeviceType.X86_64) ||
        rel.getConvention() != PelagoRel.CONVENTION){
      return null;
    }

    final PelagoRouter router = PelagoRouter.create(rel, distribution);

    RelNode newRel = planner.register(router, rel);
    RelTraitSet traitSet = rel.getTraitSet().replace(distribution);
    if (!newRel.getTraitSet().equals(traitSet)) {
      newRel = planner.changeTraits(newRel, traitSet);
    }

    return newRel;

  }

  @Override public boolean canConvert(RelOptPlanner planner, RelHomDistribution fromTrait,
      RelHomDistribution toTrait) {
    return true;
  }

  @Override public RelHomDistribution getDefault() {
    return RelHomDistribution.SINGLE;
  }
}

// End RelDeviceTypeTraitDef.java
