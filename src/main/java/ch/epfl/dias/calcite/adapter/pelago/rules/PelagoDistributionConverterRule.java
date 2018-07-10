package ch.epfl.dias.calcite.adapter.pelago.rules;

import ch.epfl.dias.calcite.adapter.pelago.PelagoRel;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoToEnumerableConverter;
import com.google.common.base.Predicates;
import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.core.Exchange;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.logical.LogicalExchange;
import org.apache.calcite.tools.RelBuilderFactory;

public class PelagoDistributionConverterRule extends ConverterRule {
//    public static final ConverterRule BRDCST_INSTANCE =
//            new PelagoDistributionConverterRule(RelDistributions.BROADCAST_DISTRIBUTED, RelDistributions.ANY            , PelagoRelFactories.PELAGO_BUILDER);
    public static final ConverterRule BRDCST_INSTANCE2 =
        new PelagoDistributionConverterRule(RelDistributions.BROADCAST_DISTRIBUTED    , RelDistributions.ANY            , PelagoRelFactories.PELAGO_BUILDER);
//    public static final ConverterRule SEQNTL_INSTANCE =
//            new PelagoDistributionConverterRule(RelDistributions.SINGLETON            , RelDistributions.RANDOM_DISTRIBUTED   , PelagoRelFactories.PELAGO_BUILDER);
    public static final ConverterRule SEQNTL_INSTANCE2 =
        new PelagoDistributionConverterRule(RelDistributions.SINGLETON            , RelDistributions.ANY   , PelagoRelFactories.PELAGO_BUILDER);
    public static final ConverterRule RANDOM_INSTANCE =
            new PelagoDistributionConverterRule(RelDistributions.RANDOM_DISTRIBUTED   , RelDistributions.SINGLETON            , PelagoRelFactories.PELAGO_BUILDER);

    private final RelDistribution distribution;

    /**
     * Creates a PelagoDistributionConverterRule.
     *
     * @param relBuilderFactory Builder for relational expressions
     */
    public PelagoDistributionConverterRule(RelDistribution distribution, RelDistribution from_distribution, RelBuilderFactory relBuilderFactory) {
        super(RelNode.class, from_distribution, distribution,"PelagoDistributionConverterRule" + distribution + from_distribution);
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
//            System.out.println(distribution + " " + rel.getTraitSet() + " " + rel);
        return PelagoRouter.create(convert(rel, PelagoRel.CONVENTION), distribution);
    }

    public boolean matches(RelOptRuleCall call) {
//        if (call.rel(0).getConvention() != PelagoRel.CONVENTION) return false;
        return true;
//        return !call.rel(0).getTraitSet().satisfies(RelTraitSet.createEmpty().plus(distribution));
    }

}
