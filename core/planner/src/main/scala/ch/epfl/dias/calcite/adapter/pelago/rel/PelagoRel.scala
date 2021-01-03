package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.traits._
import ch.epfl.dias.emitter.Binding
import org.apache.calcite.plan.{Convention, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.util.Pair
import org.json4s.JValue

object PelagoRel {
  /** Calling convention for relational operations that occur in Pelago. */
    val CONVENTION = PelagoConvention.INSTANCE

  object PelagoConvention {
    val INSTANCE = new PelagoRel.PelagoConvention("Pelago", classOf[PelagoRel])
  }

  class PelagoConvention private(val name: String, val relClass: Class[_ <: RelNode]) extends Convention.Impl(name, relClass) {
    override def useAbstractConvertersForConversion(fromTraits: RelTraitSet, toTraits: RelTraitSet): Boolean = {
      if (!fromTraits.containsIfApplicable(CONVENTION)) return false
      if (!toTraits.containsIfApplicable(CONVENTION)) return false
      //
      if (fromTraits.contains(RelComputeDevice.X86_64NVPTX)) return false
      if (toTraits.contains(RelComputeDevice.X86_64NVPTX)) return false
      var foundOne = false
      var cnt = 0
      val s1d = fromTraits.getTrait(RelSplitPointTraitDef.INSTANCE)
      val s1h = fromTraits.getTrait(RelHetDistributionTraitDef.INSTANCE)
      val s2d = toTraits.getTrait(RelSplitPointTraitDef.INSTANCE)
      val s2h = toTraits.getTrait(RelHetDistributionTraitDef.INSTANCE)
      if (toTraits.containsIfApplicable(RelDeviceType.NVPTX) && toTraits.contains(RelComputeDevice.X86_64)) return false
      if (fromTraits.containsIfApplicable(RelDeviceType.NVPTX) && fromTraits.contains(RelComputeDevice.X86_64)) return false
      if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.X86_64) && fromTraits.contains(RelComputeDevice.NVPTX)) return false
      if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.X86_64) && fromTraits.contains(RelComputeDevice.NONE)) return false
      if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.NVPTX) && fromTraits.contains(RelComputeDevice.X86_64)) return false
      if (fromTraits.containsIfApplicable(RelPacking.UnPckd) && fromTraits.containsIfApplicable(RelDeviceType.NVPTX) && fromTraits.contains(RelComputeDevice.NONE)) return false
      if (toTraits.containsIfApplicable(RelPacking.UnPckd) && toTraits.containsIfApplicable(RelDeviceType.X86_64) && toTraits.contains(RelComputeDevice.NONE)) return false
      //            if (toTraits.containsIfApplicable(RelPacking.UnPckd) && toTraits.containsIfApplicable(RelDeviceType.X86_64) && toTraits.contains(RelComputeDevice.NVPTX)) return false;
      ////            if (toTraits.containsIfApplicable(RelPacking.UnPckd) && toTraits.containsIfApplicable(RelDeviceType.X86_64) && toTraits.containsIfApplicable(RelComputeDevice.NVPTX)) return false;
      if (s1d != s2d && (fromTraits.containsIfApplicable(RelPacking.Packed) != toTraits.containsIfApplicable(RelPacking.Packed))) return false
      if (s2h != null && s2d != null) {
        if ((s2d eq RelSplitPoint.NONE) && (s2h ne RelHetDistribution.SINGLETON)) return false
        if ((s2d ne RelSplitPoint.NONE) && (s2h eq RelHetDistribution.SINGLETON)) return false
      }
      if (s1h != null && s1d != null) {
        if ((s1d eq RelSplitPoint.NONE) && (s1h ne RelHetDistribution.SINGLETON)) return false
        if ((s1d ne RelSplitPoint.NONE) && (s1h eq RelHetDistribution.SINGLETON)) return false
      }
//      if ((s1d != null) && (s2d != null) && (s1d ne RelSplitPoint.NONE) && (s2d ne RelSplitPoint.NONE)) {
//        if (s1d.point.subsetOf(s2d.point)) { //                    if (s1d.point().asJaa.diff(s2d.point().).size() > 1)
//          if (s1d.point.diff(s2d.point).size > 1) {
//            return false
//          }
//        } else if (s2d.point.subsetOf(s1d.point)) {
//          if (s2d.point.diff(s1d.point).size > 1) {
//            return false
//          }
//        } else {
//          return false;
//        }
//      }
//      if (s1d != s2d && (s1d ne RelSplitPoint.NONE) && (s2d ne RelSplitPoint.NONE)) return false;
      //            if (!(
      //                fromTraits.containsIfApplicable(RelSplitPoint.NONE()) ||
      //                    toTraits.containsIfApplicable(RelSplitPoint.NONE()))) return false;
      //            if (!fromTraits.containsIfApplicable(RelHomDistribution.SINGLE) && !toTraits.containsIfApplicable(RelHomDistribution.SINGLE)) return false;
      import scala.collection.JavaConverters._
      for (pair <- Pair.zip(fromTraits, toTraits).asScala) {
        if (!pair.left.satisfies(pair.right)) { //                    // Do not count device crossing as extra conversion
          if (!pair.left.isInstanceOf[RelComputeDevice]) {
            //                    if (pair.left instanceof RelSplitPoint && (
            //                        pair.left != RelSplitPoint.NONE() &&
            //                        pair.right != RelSplitPoint.NONE()
            //                    )) return false;
            cnt += 1
            //                    if (foundOne) return false;
            foundOne = true
            //                    if (pair.left instanceof RelPacking        ) continue;
            //                    if (pair.left instanceof RelHetDistribution) continue;
            //                    if (pair.left instanceof RelHomDistribution) continue;
            ////                    return false;
          }
        }
      }
//      if (cnt > 1){
//        println((fromTraits, toTraits))
//      }
//      cnt <= 1
      true
    }
  }

}

trait PelagoRel extends RelNode {
  def implement(target: RelDeviceType): (Binding, JValue) = implement(target, "subset" + getDigest)

  def implement(target: RelDeviceType, alias: String): (Binding, JValue)

  def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost
}