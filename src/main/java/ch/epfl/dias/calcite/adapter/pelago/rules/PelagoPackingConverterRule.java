package ch.epfl.dias.calcite.adapter.pelago.rules;

import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.tools.RelBuilderFactory;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRel;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;
import ch.epfl.dias.calcite.adapter.pelago.RelPackingTraitDef;


public class PelagoPackingConverterRule extends ConverterRule {
    public static final ConverterRule TO_PACKED_INSTANCE  =
            new PelagoPackingConverterRule(RelPacking.UnPckd , RelPacking.Packed, PelagoRelFactories.PELAGO_BUILDER);
    public static final ConverterRule TO_UNPCKD_INSTANCE =
            new PelagoPackingConverterRule(RelPacking.Packed , RelPacking.UnPckd, PelagoRelFactories.PELAGO_BUILDER);

    private final RelPacking target_packing;
    private final RelPacking from_packing  ;

    /**
     * Creates a PelagoPackingConverterRule.
     *
     * @param relBuilderFactory Builder for relational expressions
     */
    public PelagoPackingConverterRule(RelPacking target_packing, RelPacking from_packing, RelBuilderFactory relBuilderFactory) {
        super(RelNode.class, from_packing, target_packing,"PelagoDeviceTypeConverterRule" + from_packing.toString() + "_2_" + target_packing.toString());
        this.target_packing = target_packing;
        this.from_packing   = from_packing  ;
    }

    public RelNode convert(RelNode rel) {
        return RelPackingTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), convert(rel, PelagoRel.CONVENTION), target_packing, true);
    }

    public boolean matches(RelOptRuleCall call) {
        if (call.rel(0).getConvention() != PelagoRel.CONVENTION) return false;
        return true;
    }
}
