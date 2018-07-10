package ch.epfl.dias.calcite.adapter.pelago.rules;

//import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRel;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
//import ch.epfl.dias.calcite.adapter.pelago.trait.RelDeviceType;
//import ch.epfl.dias.calcite.adapter.pelago.trait.RelDeviceTypeTraitDef;
import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.tools.RelBuilderFactory;

public class PelagoDeviceTypeConverterRule extends ConverterRule {
    public static final ConverterRule TO_NVPTX_INSTANCE  =
            new PelagoDeviceTypeConverterRule(RelDeviceType.NVPTX , RelDeviceType.X86_64, PelagoRelFactories.PELAGO_BUILDER);
    public static final ConverterRule TO_x86_64_INSTANCE =
            new PelagoDeviceTypeConverterRule(RelDeviceType.X86_64, RelDeviceType.NVPTX , PelagoRelFactories.PELAGO_BUILDER);

    private final RelDeviceType target_device;
    private final RelDeviceType from_device  ;

    /**
     * Creates a PelagoDistributionConverterRule.
     *
     * @param relBuilderFactory Builder for relational expressions
     */
    public PelagoDeviceTypeConverterRule(RelDeviceType target_device, RelDeviceType from_device, RelBuilderFactory relBuilderFactory) {
        super(RelNode.class, from_device, target_device,"PelagoDeviceTypeConverterRule" + from_device.toString() + "_2_" + target_device.toString());
        this.target_device = target_device;
        this.from_device   = from_device  ;
    }

    public RelNode convert(RelNode rel) {
////        if (distribution == RelDistributions.RANDOM_DISTRIBUTED) {
////            System.out.println(distribution + " " + rel.getTraitSet() + " " + rel);
////        }
//        RelNode tmp = RelDeviceTypeTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), rel, target_device, true);
////        if (distribution == RelDistributions.RANDOM_DISTRIBUTED) {
////            System.out.println(tmp);
////        }
        return PelagoDeviceCross.create(convert(rel, PelagoRel.CONVENTION), target_device);
//        return LogicalDeviceCross.create(rel, target_device);
//        return PelagoDeviceCross.create(rel, target_device);
    }

    public boolean matches(RelOptRuleCall call) {
        if (call.rel(0).getConvention() != PelagoRel.CONVENTION) return false;
        return true;
//        return !call.rel(0).getTraitSet().satisfies(RelTraitSet.createEmpty().plus(target_device));
    }
}
