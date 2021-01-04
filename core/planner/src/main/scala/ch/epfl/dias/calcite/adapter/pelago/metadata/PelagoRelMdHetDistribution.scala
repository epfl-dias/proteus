/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
 */

package ch.epfl.dias.calcite.adapter.pelago.metadata

import ch.epfl.dias.calcite.adapter.pelago.rel.{
  PelagoSplit,
  PelagoTableScan,
  PelagoUnion,
  PelagoUnpack
}
import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelHetDistribution,
  RelHetDistributionTraitDef
}
import org.apache.calcite.plan.volcano.RelSubset
import org.apache.calcite.rel.{BiRel, RelNode, SingleRel}
import org.apache.calcite.rel.metadata.{
  MetadataDef,
  MetadataHandler,
  ReflectiveRelMetadataProvider,
  RelMetadataProvider,
  RelMetadataQuery
}

object PelagoRelMdHetDistribution {
  private val INSTANCE = new PelagoRelMdHetDistribution
  val SOURCE: RelMetadataProvider =
    ReflectiveRelMetadataProvider.reflectiveSource(
      HetDistribution.method,
      PelagoRelMdHetDistribution.INSTANCE
    )
}
class PelagoRelMdHetDistribution extends MetadataHandler[HetDistribution] {
  override def getDef: MetadataDef[HetDistribution] = HetDistribution.DEF

  def hetDistribution(rel: RelNode, mq: RelMetadataQuery): RelHetDistribution =
    RelHetDistributionTraitDef.INSTANCE.getDefault

  def hetDistribution(
      rel: RelSubset,
      mq: RelMetadataQuery
  ): RelHetDistribution =
    rel.getTraitSet.getTrait(RelHetDistributionTraitDef.INSTANCE)

  def hetDistribution(
      rel: SingleRel,
      mq: RelMetadataQuery
  ): RelHetDistribution =
    mq.asInstanceOf[PelagoRelMetadataQuery].hetDistribution(rel.getInput)

  def hetDistribution(
      rel: PelagoUnpack,
      mq: RelMetadataQuery
  ): RelHetDistribution =
    mq.asInstanceOf[PelagoRelMetadataQuery].hetDistribution(rel.getInput)

  def hetDistribution(rel: BiRel, mq: RelMetadataQuery): RelHetDistribution =
    mq.asInstanceOf[PelagoRelMetadataQuery].hetDistribution(rel.getRight)

  def hetDistribution(
      scan: PelagoTableScan,
      mq: RelMetadataQuery
  ): RelHetDistribution =
    RelHetDistribution.SINGLETON

  def hetDistribution(
      split: PelagoSplit,
      mq: RelMetadataQuery
  ): RelHetDistribution =
    split.hetdistribution

  def hetDistribution(
      union: PelagoUnion,
      mq: RelMetadataQuery
  ): RelHetDistribution =
    RelHetDistribution.SINGLETON
}
