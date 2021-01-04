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

package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.traits._
import ch.epfl.dias.emitter.Binding
import org.apache.calcite.plan.{
  RelOptCluster,
  RelOptCost,
  RelOptPlanner,
  RelTraitSet
}
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.{
  CyclicMetadataException,
  RelMetadataQuery
}
import org.apache.calcite.rel.{RelNode, _}
import org.json4s.JsonDSL._
import org.json4s._

import scala.collection.mutable

class PelagoSplit protected (
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    val hetdistribution: RelHetDistribution,
    val splitId: Long
) extends PelagoRouter(
      cluster,
      traitSet,
      input,
      traitSet.getTrait(RelHomDistributionTraitDef.INSTANCE)
    )
    with Converter {
//  assert(splitId == RelSplitPoint.getOrCreateId(input).get)
  assert(getConvention eq PelagoRel.CONVENTION)
  assert(getConvention eq input.getConvention)

  override def copy(traitSet: RelTraitSet, input: RelNode): PelagoSplit =
    PelagoSplit.create(input, hetdistribution, splitId)

//  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
//    val bcost = super.computeBaseSelfCost(planner, mq)
//
//    return planner.getCostFactory.makeCost(bcost.getRows / 2, bcost.getCpu, bcost.getIo)
//    //    planner.getCostFactory.makeZeroCost()
//  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = {
    var rc: Double =
      try {
        mq.getRowCount(getInput)
      } catch {
        case _: CyclicMetadataException => 1.0e50
      }
    if (
      (hetdistribution eq RelHetDistribution.SPLIT_BRDCST) && input.getTraitSet
        .contains(RelHetDistribution.SPLIT)
    ) rc = rc * 2
    else if (hetdistribution eq RelHetDistribution.SPLIT_BRDCST) rc = rc
    else if (hetdistribution eq RelHetDistribution.SPLIT) rc = rc / 2.0
    else if (hetdistribution eq RelHetDistribution.SINGLETON) rc = rc * 2.0
    rc
  }

  override def computeBaseSelfCost(
      planner: RelOptPlanner,
      mq: RelMetadataQuery
  ): RelOptCost = {
    //    if (traitSet.containsIfApplicable(RelPacking.UnPckd)) return planner.getCostFactory.makeHugeCost()
    val rf =
      1e6 * (if (hetdistribution == RelHetDistribution.SPLIT_BRDCST) 10 else 1)
    var base = super.computeBaseSelfCost(planner, mq)
    //    if (getDistribution.getType eq RelDistribution.Type.HASH_DISTRIBUTED) base = base.multiplyBy(80)
    planner.getCostFactory.makeCost(base.getRows, base.getCpu * rf, base.getIo)
    //    planner.getCostFactory.makeZeroCost()
  }

  override def explainTerms(pw: RelWriter): RelWriter =
    super
      .explainTerms(pw)
      .item("het_distribution", hetdistribution.toString)
      .item("splitId", splitId)

//  override def estimateRowCount(mq: RelMetadataQuery): Double = super.estimateRowCount(mq)/2

  override def implement(
      target: RelDeviceType,
      alias: String
  ): (Binding, JValue) = {
    //    assert(getTraitSet.containsIfApplicable(RelPacking.UnPckd) || (target != null))
    assert(target != null)
    assert(hetdistribution != RelHetDistribution.SINGLETON)

    val header = ("operator", "split") ~ ("split_id", splitId)

    val binding = PelagoSplit.bindings.remove(splitId)
    if (binding.isDefined) {
      // TODO: if we increase the number of device types to more than two,
      //  we should user a count instead of null/not null
      return (binding.get, header)
    }

    val child = getInput.asInstanceOf[PelagoRel].implement(null, alias)
    val childBinding: Binding = child._1
    var childOp = child._2

    var out_dop = this.hetdistribution.getNumOfDeviceTypes

    var in_dop = input.getTraitSet
      .getTrait(RelHetDistributionTraitDef.INSTANCE)
      .getNumOfDeviceTypes

    val policy: JObject = {
      if (hetdistribution eq RelHetDistribution.SPLIT_BRDCST) {
        if (getTraitSet.containsIfApplicable(RelPacking.Packed)) {
          childOp = ("operator", "mem-broadcast-device") ~
            ("num_of_targets", out_dop) ~
            ("input", child._2) ~
            ("to_cpu", true) ~
            ("always_share", true)
        }

        (
          "target",
          ("expression", "recordProjection") ~
            ("e",
            ("expression", "argument")) ~
            ("attribute",
            ("relName", childBinding.rel.getPelagoRelName) ~
              ("attrName", "__broadcastTarget"))
        )
      } else if (hetdistribution == RelHetDistribution.SPLIT) {
        JObject()
      } else {
        // else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        assert(assertion = false, "translation not implemented")
        ("operator", "hash-split")
      }
    }

    val json = header ~
      ("numOfParents", out_dop) ~
      ("producers", in_dop) ~
      ("slack", 8) ~
      policy ~
      ("input", childOp)

    PelagoSplit.bindings(splitId) = childBinding
    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }
}

object PelagoSplit {
  def create(input: RelNode, distribution: RelDistribution): PelagoSplit = {
    assert(false);
    return null;
  }

  val bindings: mutable.Map[Long, Binding] =
    collection.mutable.Map[Long, Binding]();

  def create(input: RelNode, distribution: RelHetDistribution): PelagoSplit = {
    val splitId = RelSplitPoint.getOrCreateId(input)
    assert(splitId.nonEmpty)
    val traitSet = input.getTraitSet
      .replace(PelagoRel.CONVENTION)
      .replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.NONE)
      .replaceIf(
        RelSplitPointTraitDef.INSTANCE,
        () => RelSplitPoint.of(splitId.get)
      )
    new PelagoSplit(
      input.getCluster,
      traitSet,
      input,
      distribution,
      splitId.get
    )
  }

  def create(
      input: RelNode,
      distribution: RelHetDistribution,
      _splitId: Long
  ): PelagoSplit = {
    val splitId = RelSplitPoint.getOrCreateId(input).getOrElse(_splitId)
    val traitSet = input.getTraitSet
      .replace(PelagoRel.CONVENTION)
      .replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.NONE)
      .replaceIf(
        RelSplitPointTraitDef.INSTANCE,
        () => RelSplitPoint.of(splitId)
      )
    new PelagoSplit(input.getCluster, traitSet, input, distribution, splitId)
  }
}
