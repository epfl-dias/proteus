package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.tools.RelBuilderFactory;

import ch.epfl.dias.calcite.adapter.pelago.PelagoRel;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelDistributions;
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMdHetDistribution;

public class PelagoHetDistributionConverterRule extends ConverterRule {
//    public static final ConverterRule BRDCST_INSTANCE =
//            new PelagoDistributionConverterRule(RelDistributions.BROADCAST_DISTRIBUTED, RelDistributions.ANY            , PelagoRelFactories.PELAGO_BUILDER);
    public static final ConverterRule BRDCST_INSTANCE =
        new PelagoHetDistributionConverterRule(RelHetDistribution.SPLIT_BRDCST, PelagoRelFactories.PELAGO_BUILDER);
    public static final ConverterRule RANDOM_INSTANCE =
        new PelagoHetDistributionConverterRule(RelHetDistribution.SPLIT       , PelagoRelFactories.PELAGO_BUILDER);

    private final RelHetDistribution distribution;

    /**
     * Creates a PelagoDistributionConverterRule.
     *
     * @param relBuilderFactory Builder for relational expressions
     */
    public PelagoHetDistributionConverterRule(RelHetDistribution distribution, RelBuilderFactory relBuilderFactory) {
        super(RelNode.class, RelHetDistribution.SINGLETON, distribution,"PelagoHetDistributionConverterRule" + distribution);
        this.distribution = distribution;
    }

    @Override public RelNode convert(RelNode rel) {
//        if (distribution == RelDistributions.RANDOM_DISTRIBUTED) {
//            System.out.println(distribution + " " + rel.getTraitSet() + " " + rel);
//        }
//        RelNode tmp = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), rel, distribution, true);
//        if (distribution == RelDistributions.RANDOM_DISTRIBUTED) {
//            System.out.println(tmp);
//        }
        RelTraitSet traitSet = rel.getTraitSet().replace(PelagoRel.CONVENTION).replace(RelDeviceType.X86_64);
//            System.out.println(distribution + " " + rel.getTraitSet() + " " + rel);
        return PelagoSplit.create(convert(rel, traitSet), distribution);
    }

    public boolean matches(RelOptRuleCall call) {
//        if (call.rel(0).getConvention() != PelagoRel.CONVENTION) return false;
//        return true;
        return call.rel(0).getTraitSet().containsIfApplicable(RelHetDistribution.SINGLETON) &&
            !call.rel(0).getTraitSet().containsIfApplicable(distribution);
    }

}
