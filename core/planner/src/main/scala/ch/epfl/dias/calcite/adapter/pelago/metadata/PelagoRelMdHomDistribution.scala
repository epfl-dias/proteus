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

import ch.epfl.dias.calcite.adapter.pelago.rel._
import ch.epfl.dias.calcite.adapter.pelago.traits.{
  RelHomDistribution,
  RelHomDistributionTraitDef
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
import org.apache.calcite.rex.RexNode

import java.util

object PelagoRelMdHomDistribution {
  private val INSTANCE = new PelagoRelMdHomDistribution
  val SOURCE: RelMetadataProvider =
    ReflectiveRelMetadataProvider.reflectiveSource(
      HomDistribution.method,
      PelagoRelMdHomDistribution.INSTANCE
    )

  def project(
      mq: RelMetadataQuery,
      input: RelNode,
      projects: util.List[_ <: RexNode]
  ): RelHomDistribution = { //    return mq.distribution(input);
//    Mappings.TargetMapping mapping = Project.getPartialMapping(input.getRowType().getFieldCount(), projects);
//    return inputDistribution.apply(mapping);
    mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(input)
  }
}

class PelagoRelMdHomDistribution extends MetadataHandler[HomDistribution] {
  override def getDef: MetadataDef[HomDistribution] = HomDistribution.DEF

  def homDistribution(
      rel: RelSubset,
      mq: RelMetadataQuery
  ): RelHomDistribution =
    rel.getTraitSet.getTrait(RelHomDistributionTraitDef.INSTANCE)

  def homDistribution(
      rel: SingleRel,
      mq: RelMetadataQuery
  ): RelHomDistribution =
    mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(rel.getInput)

  def homDistribution(
      rel: PelagoUnpack,
      mq: RelMetadataQuery
  ): RelHomDistribution =
    mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(rel.getInput)

  def homDistribution(rel: BiRel, mq: RelMetadataQuery): RelHomDistribution =
    mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(rel.getRight)

  def homDistribution(
      split: PelagoSplit,
      mq: RelMetadataQuery
  ): RelHomDistribution =
    split.homdistribution

  def homDistribution(
      union: PelagoUnion,
      mq: RelMetadataQuery
  ): RelHomDistribution =
    RelHomDistribution.SINGLE

  def homDistribution(
      mod: PelagoTableModify,
      mq: RelMetadataQuery
  ): RelHomDistribution = {
    val ret =
      mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(mod.getInput)
    assert(ret eq RelHomDistribution.SINGLE)
    ret
  }

  def homDistribution(
      rel: RelNode,
      mq: RelMetadataQuery
  ): RelHomDistribution = {
    val dtype =
      rel.getTraitSet.getTrait(
        RelHomDistributionTraitDef.INSTANCE
      ) //TODO: is this safe ? or can it cause an inf loop?
    if (dtype != null) return dtype
    RelHomDistribution.SINGLE
  }

  def homDistribution(
      router: PelagoRouter,
      mq: RelMetadataQuery
  ): RelHomDistribution = { //    System.out.println(scan.getDistribution());
    router.getHomDistribution
  }

  def homDistribution(
      scan: PelagoTableScan,
      mq: RelMetadataQuery
  ): RelHomDistribution =
    scan.getHomDistribution

  def homDistribution(
      devcross: PelagoDeviceCross,
      mq: RelMetadataQuery
  ): RelHomDistribution = { //    System.out.println("asdasd");
    mq.asInstanceOf[PelagoRelMetadataQuery].homDistribution(devcross.getInput)
  }
}
