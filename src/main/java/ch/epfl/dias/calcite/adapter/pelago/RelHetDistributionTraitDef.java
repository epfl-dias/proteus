package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;

import java.util.List;

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
//      input = planner.register(planner.changeTraits(rel, inptraitSet), rel);
      return null;
    }

    PelagoRel router;
    if (distribution == RelHetDistribution.SPLIT || distribution == RelHetDistribution.SPLIT_BRDCST) {
      if (!input.getTraitSet().contains(RelHetDistribution.SINGLETON)) return null;
      router = PelagoSplit.create(input, distribution);
    } else {
      if (!input.getTraitSet().contains(RelHetDistribution.SPLIT)) return null;
      RelTraitSet c = input.getTraitSet().replace(RelComputeDevice.X86_64);
      RelTraitSet g = input.getTraitSet().replace(RelComputeDevice.NVPTX);
      RelNode ing = input;
      RelNode inc = input;
      if (!ing.getTraitSet().equals(g)){
        ing = planner.changeTraits(ing, g);
      }
      if (!inc.getTraitSet().equals(c)){
        inc = planner.changeTraits(inc, c);
      }
      router = PelagoUnion.create(List.of(
            inc,
            ing
          ), true);
    }

    RelNode newRel = planner.register(router, rel);
    if (!newRel.getTraitSet().equals(traitSet)) {
      newRel = planner.changeTraits(newRel, traitSet);
    }

    return newRel;
  }

  @Override public boolean canConvert(RelOptPlanner planner, RelHetDistribution fromTrait,
      RelHetDistribution toTrait) {
    return toTrait != fromTrait && fromTrait != RelHetDistribution.SPLIT_BRDCST;
  }

  @Override public RelHetDistribution getDefault() {
    return RelHetDistribution.SINGLETON;
  }
}

// End RelDeviceTypeTraitDef.java
