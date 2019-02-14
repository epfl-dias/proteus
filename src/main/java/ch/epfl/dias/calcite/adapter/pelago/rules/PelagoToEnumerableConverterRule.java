package ch.epfl.dias.calcite.adapter.pelago.rules;

import ch.epfl.dias.calcite.adapter.pelago.PelagoRel;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.PelagoToEnumerableConverter;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;

import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.tools.RelBuilderFactory;

/**
 * Rule to convert a relational expression from
 * {@link PelagoRel#CONVENTION} to {@link EnumerableConvention}.
 */
public class PelagoToEnumerableConverterRule extends ConverterRule {
    public static final ConverterRule INSTANCE =
            new PelagoToEnumerableConverterRule(PelagoRelFactories.PELAGO_BUILDER);

    /**
     * Creates a CassandraToEnumerableConverterRule.
     *
     * @param relBuilderFactory Builder for relational expressions
     */
    public PelagoToEnumerableConverterRule(
            RelBuilderFactory relBuilderFactory) {
        super(RelNode.class, PelagoRel.CONVENTION, EnumerableConvention.INSTANCE,
                "PelagoToEnumerableConverterRule");
    }

    @Override public RelNode convert(RelNode rel) {
//        RelTraitSet newTraitSet = rel.getTraitSet().replace(getOutConvention()); //.replace(RelDeviceType.ANY);
//        RelNode inp = rel;
//        RelNode inp = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), rel, RelDistributions.SINGLETON, true);
//        RelNode inp = LogicalExchange.create(rel, RelDistributions.SINGLETON);
//        System.out.println(inp.getTraitSet());

        RelTraitSet traitSet = rel.getTraitSet().replace(PelagoRel.CONVENTION)
            .replace(RelHomDistribution.SINGLE)
            .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64);

//        RelNode inp = rel;//convert(convert(rel, RelDistributions.SINGLETON), RelDeviceType.X86_64); //Convert to sequential
        RelNode inp = convert(rel, traitSet);

        RelNode tmp = PelagoToEnumerableConverter.create(convert(inp, RelHetDistribution.SINGLETON));
        return tmp;
    }

    public boolean matches(RelOptRuleCall call) {
//        return true;
//        if (!call.rel(0).getTraitSet().satisfies(RelTraitSet.createEmpty().plus(RelDistributions.SINGLETON))) return false;
//        if (!call.rel(0).getTraitSet().contains(RelDeviceType.X86_64)) return false;
//        if (call.rel(0).getTraitSet().containsIfApplicable(PelagoRel.CONVENTION)) return false;
        return call.rel(0).getTraitSet().containsIfApplicable(RelHetDistribution.SINGLETON);
    }
}