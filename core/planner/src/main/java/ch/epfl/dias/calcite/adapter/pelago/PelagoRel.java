//package ch.epfl.dias.calcite.adapter.pelago;
//
//import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery;
//import ch.epfl.dias.emitter.Binding;
//import org.apache.calcite.plan.Convention;
//import org.apache.calcite.plan.RelOptCluster;
//import org.apache.calcite.plan.RelOptCost;
//import org.apache.calcite.plan.RelOptPlanner;
//import org.apache.calcite.plan.RelTrait;
//import org.apache.calcite.plan.RelTraitSet;
//import org.apache.calcite.rel.RelDistribution;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.RelWriter;
//import org.apache.calcite.rel.metadata.RelMetadataQuery;
//import org.apache.calcite.util.Pair;
//
//import com.google.common.collect.ImmutableList;
//
//import org.json4s.JsonAST;
//import scala.Tuple2;
//
//public interface PelagoRel extends RelNode {
//    /** Calling convention for relational operations that occur in Pelago. */
//    Convention CONVENTION = PelagoConvention.INSTANCE;
//
//    default Tuple2<Binding, JsonAST.JValue> implement(RelDeviceType target){
//        return implement(target, "subset" + getDigest());
//    }
//
//    Tuple2<Binding, JsonAST.JValue> implement(RelDeviceType target, String alias);
//
//    RelOptCost computeBaseSelfCost(RelOptPlanner planner, RelMetadataQuery mq);
//
//    class PelagoConvention extends Convention.Impl{
//        public static final PelagoConvention INSTANCE = new PelagoConvention("Pelago", PelagoRel.class);
//
//        private PelagoConvention(final String name, final Class<? extends RelNode> relClass) {
//            super(name, relClass);
//        }
//
//        public boolean useAbstractConvertersForConversion(RelTraitSet fromTraits, RelTraitSet toTraits) {
//            if (!fromTraits.containsIfApplicable(CONVENTION)) return false;
//            if (!toTraits.containsIfApplicable(CONVENTION)) return false;
////
////
//            if (fromTraits.contains(RelComputeDevice.X86_64NVPTX)) return false;
//            if (toTraits.contains(RelComputeDevice.X86_64NVPTX)) return false;
//            boolean foundOne = false;
//            int cnt = 0;
//
//            var s1d = fromTraits.getTrait(RelSplitPointTraitDef.INSTANCE);
//            var s1h = fromTraits.getTrait(RelHetDistributionTraitDef.INSTANCE);
//            var s2d = toTraits.getTrait(RelSplitPointTraitDef.INSTANCE);
//            var s2h = toTraits.getTrait(RelHetDistributionTraitDef.INSTANCE);
//
//            if (toTraits.containsIfApplicable(RelDeviceType.NVPTX) && toTraits.contains(RelComputeDevice.X86_64)) return false;
//            if (fromTraits.containsIfApplicable(RelDeviceType.NVPTX) && fromTraits.contains(RelComputeDevice.X86_64)) return false;
//            if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.X86_64) && fromTraits.contains(RelComputeDevice.NVPTX)) return false;
//            if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.X86_64) && fromTraits.contains(RelComputeDevice.NONE)) return false;
//            if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.NVPTX) && fromTraits.contains(RelComputeDevice.X86_64)) return false;
//            if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.NVPTX) && fromTraits.contains(RelComputeDevice.NONE)) return false;
//            if (toTraits.containsIfApplicable(RelPacking.UnPckd) && toTraits.containsIfApplicable(RelDeviceType.X86_64) && toTraits.contains(RelComputeDevice.NONE)) return false;
////            if (toTraits.containsIfApplicable(RelPacking.UnPckd) && toTraits.containsIfApplicable(RelDeviceType.X86_64) && toTraits.contains(RelComputeDevice.NVPTX)) return false;
//////            if (toTraits.containsIfApplicable(RelPacking.UnPckd) && toTraits.containsIfApplicable(RelDeviceType.X86_64) && toTraits.containsIfApplicable(RelComputeDevice.NVPTX)) return false;
//
//            if (s2h != null && s2d != null) {
//                if (s2d == RelSplitPoint.NONE() && s2h != RelHetDistribution.SINGLETON) return false;
//                if (s2d != RelSplitPoint.NONE() && s2h == RelHetDistribution.SINGLETON) return false;
//            }
//
//            if (s1h != null && s1d != null) {
//                if (s1d == RelSplitPoint.NONE() && s1h != RelHetDistribution.SINGLETON) return false;
//                if (s1d != RelSplitPoint.NONE() && s1h == RelHetDistribution.SINGLETON) return false;
//            }
//            if (s1d != null && s2d != null){
//                if (s1d.point().subsetOf(s2d.point())) {
////                    if (s1d.point().asJaa.diff(s2d.point().).size() > 1)
//                        return false;
//                } else if (s2d.point().subsetOf(s1d.point())) {
//                    if (s2d.point().diff(s1d.point()).size() > 1) return false;
//                } else {
//                    return false;
//                }
//            }
//
////            if (!(
////                fromTraits.containsIfApplicable(RelSplitPoint.NONE()) ||
////                    toTraits.containsIfApplicable(RelSplitPoint.NONE()))) return false;
//
////            if (!fromTraits.containsIfApplicable(RelHomDistribution.SINGLE) && !toTraits.containsIfApplicable(RelHomDistribution.SINGLE)) return false;
//
//            for (Pair<RelTrait, RelTrait> pair : Pair.zip(fromTraits, toTraits)) {
//                if (!pair.left.satisfies(pair.right)) {
////                    // Do not count device crossing as extra conversion
//                    if (pair.left instanceof RelComputeDevice     ) continue;
////                    if (pair.left instanceof RelSplitPoint && (
////                        pair.left != RelSplitPoint.NONE() &&
////                        pair.right != RelSplitPoint.NONE()
////                    )) return false;
//                    ++cnt;
////
////                    if (foundOne) return false;
//                    foundOne = true;
////
////                    if (pair.left instanceof RelPacking        ) continue;
////                    if (pair.left instanceof RelHetDistribution) continue;
////                    if (pair.left instanceof RelHomDistribution) continue;
//////                    return false;
//                }
//            }
//            return true;
//        }
//    }
//}
//

