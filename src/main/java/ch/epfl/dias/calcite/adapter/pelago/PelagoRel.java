package ch.epfl.dias.calcite.adapter.pelago;

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery;
import ch.epfl.dias.emitter.Binding;
import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.util.Pair;

import org.json4s.JsonAST;
import scala.Tuple2;

public interface PelagoRel extends RelNode {
    /** Calling convention for relational operations that occur in Pelago. */
    Convention CONVENTION = PelagoConvention.INSTANCE;

    default Tuple2<Binding, JsonAST.JValue> implement(RelDeviceType target){
        return implement(target, "subset" + getDigest());
    }

    Tuple2<Binding, JsonAST.JValue> implement(RelDeviceType target, String alias);

    RelOptCost computeBaseSelfCost(RelOptPlanner planner, RelMetadataQuery mq);

    class PelagoConvention extends Convention.Impl{
        public static final PelagoConvention INSTANCE = new PelagoConvention("Pelago", PelagoRel.class);

        private PelagoConvention(final String name, final Class<? extends RelNode> relClass) {
            super(name, relClass);
        }

        public boolean useAbstractConvertersForConversion(RelTraitSet fromTraits, RelTraitSet toTraits) {
            if (!fromTraits.containsIfApplicable(CONVENTION)) return false;

            for (Pair<RelTrait, RelTrait> pair : Pair.zip(fromTraits, toTraits)) {
                if (!pair.left.satisfies(pair.right)) {
                    if (pair.left instanceof RelDeviceType     ) continue;
                    if (pair.left instanceof RelPacking        ) continue;
                    if (pair.left instanceof RelHetDistribution) continue;
                    if (pair.left instanceof RelHomDistribution) continue;
                    return false;
                }
            }
            return true;
        }
    }
}

