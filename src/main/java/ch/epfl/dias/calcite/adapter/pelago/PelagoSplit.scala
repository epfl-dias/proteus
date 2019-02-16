package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema}
import ch.epfl.dias.repl.Repl
import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel.{RelNode, _}
import org.apache.calcite.rel.convert.Converter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexInputRef
import org.json4s.JsonDSL._
import org.json4s._

import scala.collection.JavaConverters._
import ch.epfl.dias.repl.Repl

class PelagoSplit protected(cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, val hetdistribution: RelHetDistribution, val splitId: Long)
    extends PelagoRouter(cluster, traitSet, input, traitSet.getTrait(RelHomDistributionTraitDef.INSTANCE)) with Converter {
  assert(getConvention eq PelagoRel.CONVENTION)
  assert(getConvention eq input.getConvention)

  override def copy(traitSet: RelTraitSet, input: RelNode) = PelagoSplit.create(input, hetdistribution, splitId)

//  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
//    val bcost = super.computeBaseSelfCost(planner, mq)
//
//    return planner.getCostFactory.makeCost(bcost.getRows / 2, bcost.getCpu, bcost.getIo)
//    //    planner.getCostFactory.makeZeroCost()
//  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = {
    var rc = mq.getRowCount(getInput)
    if      (hetdistribution eq RelHetDistribution.SPLIT_BRDCST) rc = rc
    else if (hetdistribution eq RelHetDistribution.SPLIT       ) rc = rc / 2.0
    else if (hetdistribution eq RelHetDistribution.SINGLETON   ) rc = rc * 2.0
    rc
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    //    if (traitSet.containsIfApplicable(RelPacking.UnPckd)) return planner.getCostFactory.makeHugeCost()
    val rf = 1e6 * (if (hetdistribution == RelHetDistribution.SPLIT_BRDCST) 10 else 1)
    var base = super.computeBaseSelfCost(planner, mq)
    //    if (getDistribution.getType eq RelDistribution.Type.HASH_DISTRIBUTED) base = base.multiplyBy(80)
    planner.getCostFactory.makeCost(base.getRows, base.getCpu * rf, base.getIo)
    //    planner.getCostFactory.makeZeroCost()
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("het_distribution", hetdistribution.toString).item("splitId", splitId)

//  override def estimateRowCount(mq: RelMetadataQuery): Double = super.estimateRowCount(mq)/2

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    //    assert(getTraitSet.containsIfApplicable(RelPacking.UnPckd) || (target != null))
    assert(target != null)
    assert(hetdistribution != RelHetDistribution.SINGLETON)

    val header = ("operator", "split") ~ ("split_id", splitId)

    val binding = PelagoSplit.bindings.remove(splitId)
    if (binding.isDefined){
      // TODO: if we increase the number of device types to more than two,
      //  we should user a count instead of null/not null
      return (binding.get, header)
    }

    val child = getInput.asInstanceOf[PelagoRel].implement(null, alias)
    val childBinding: Binding = child._1
    var childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType, false, getTraitSet.containsIfApplicable(RelPacking.Packed))

    var out_dop = this.hetdistribution.getNumOfDeviceTypes()

    var in_dop = input.getTraitSet.getTrait(RelHetDistributionTraitDef.INSTANCE).getNumOfDeviceTypes

    val projs = getRowType.getFieldList.asScala.zipWithIndex.map{
      f => {
        emitExpression(RexInputRef.of(f._2, getInput.getRowType), List(childBinding)).asInstanceOf[JObject]
      }
    }

    val policy: JObject = {
      if (hetdistribution eq RelHetDistribution.SPLIT_BRDCST) {
        if (getTraitSet.containsIfApplicable(RelPacking.Packed)) {
          childOp = ("operator", "mem-broadcast-device") ~
            ("num_of_targets", out_dop) ~
            ("projections", emitSchema(childBinding.rel, getRowType, false, true)) ~
            ("input", child._2) ~
            ("to_cpu", true) ~
            ("always_share", true)
        }

        ("target",
          ("expression", "recordProjection") ~
            ("e",
              ("expression", "argument") ~
                ("argNo", -1) ~
                ("type",
                  ("type", "record") ~
                    ("relName", childBinding.rel)
                ) ~
                ("attributes", List(
                  ("relName", childBinding.rel) ~
                    ("attrName", "__broadcastTarget")
                ))
            ) ~
            ("attribute",
              ("relName", childBinding.rel) ~
                ("attrName", "__broadcastTarget")
            )
        )
      } else if (hetdistribution == RelHetDistribution.SPLIT) {
        JObject()
      } else {
        // else if (getDistribution.getType eq RelDistribution.Type.SINGLETON) {
        assert(false, "translation not implemented")
        ("operator", "hash-split")
      }
    }

    var json = header ~
      ("gpu"         , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("projections" , rowType) ~
      ("numOfParents", out_dop) ~
      ("producers"   , in_dop) ~
      ("slack"       , 8) ~
      policy ~
      ("input", childOp)

    PelagoSplit.bindings(splitId) = childBinding
    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }
}

object PelagoSplit{
  def create(input: RelNode, distribution: RelDistribution): PelagoSplit = {
    assert(false);
    return null;
  }

  var split_cnt: Long = 0;
  val bindings = collection.mutable.Map[Long, Binding]();

  def create(input: RelNode, distribution: RelHetDistribution): PelagoSplit = {
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.NONE)
    val splitId = split_cnt
    split_cnt = splitId + 1
    new PelagoSplit(input.getCluster, traitSet, input, distribution, splitId)
  }

  def create(input: RelNode, distribution: RelHetDistribution, splitId: Long): PelagoSplit = {
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION).replace(distribution)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.NONE)
    new PelagoSplit(input.getCluster, traitSet, input, distribution, splitId)
  }
}
