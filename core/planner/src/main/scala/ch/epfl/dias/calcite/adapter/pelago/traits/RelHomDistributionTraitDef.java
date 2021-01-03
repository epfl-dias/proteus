package ch.epfl.dias.calcite.adapter.pelago.traits;

import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoRel;
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoRouter;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
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

    if (rel.getConvention() != PelagoRel.CONVENTION() || !rel.getTraitSet().containsIfApplicable(RelDeviceType.X86_64)){
      return null;
    }

    RelTraitSet inptraitSet = rel.getTraitSet();
    RelTraitSet traitSet = rel.getTraitSet().replace(distribution);
    RelNode input = rel;
//    if (!rel.getTraitSet().equals(inptraitSet)) {
////      input = planner.register(planner.changeTraits(rel, inptraitSet), rel);
//      return null;
//    }

    final PelagoRouter router = PelagoRouter.create(input, distribution);

    RelNode newRel = planner.register(router, rel);
    if (!newRel.getTraitSet().equals(traitSet)) {
      newRel = planner.register(planner.changeTraits(newRel, traitSet), rel);
    }

    return newRel;

  }

  @Override public boolean canConvert(RelOptPlanner planner, RelHomDistribution fromTrait,
      RelHomDistribution toTrait) {
    if (RelHomDistribution.SINGLE == fromTrait) {
      return toTrait == RelHomDistribution.RANDOM || toTrait == RelHomDistribution.BRDCST;
    } else if (RelHomDistribution.RANDOM == fromTrait) {
      return toTrait == RelHomDistribution.SINGLE; //FIXME: RANDOM can become BRDCST but I am not sure the executor supports it
    }
    return false; // Can't convert BRDCST
  }

  @Override public RelHomDistribution getDefault() {
    return RelHomDistribution.SINGLE;
  }
}

// End RelDeviceTypeTraitDef.java
