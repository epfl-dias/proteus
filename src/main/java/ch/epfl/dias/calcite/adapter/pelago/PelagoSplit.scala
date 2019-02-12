package ch.epfl.dias.calcite.adapter.pelago

import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel.{RelNode, _}
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery

class PelagoSplit protected(cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, val hetdistribution: RelHetDistribution)
    extends PelagoRouter(cluster, traitSet, input, input.getTraitSet.getTrait(RelDistributionTraitDef.INSTANCE)) with Converter {
  assert(getConvention eq PelagoRel.CONVENTION)
  assert(getConvention eq input.getConvention)

  override def copy(traitSet: RelTraitSet, input: RelNode, ingore: RelDistribution) = PelagoSplit.create(input, hetdistribution)

//  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
//    val bcost = super.computeBaseSelfCost(planner, mq)
//
//    return planner.getCostFactory.makeCost(bcost.getRows / 2, bcost.getCpu, bcost.getIo)
//    //    planner.getCostFactory.makeZeroCost()
//  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("het_distribution", hetdistribution.toString)

  override def estimateRowCount(mq: RelMetadataQuery): Double = super.estimateRowCount(mq)/2
}

object PelagoSplit{
  def create(input: RelNode, distribution: RelDistribution): PelagoSplit = {
    assert(false);
    return null;
  }

  def create(input: RelNode, distribution: RelHetDistribution): PelagoSplit = {
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.NONE)
    new PelagoSplit(input.getCluster, traitSet, input, distribution)
  }
}
