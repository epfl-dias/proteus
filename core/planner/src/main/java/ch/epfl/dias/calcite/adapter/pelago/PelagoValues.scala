package ch.epfl.dias.calcite.adapter.pelago

import java.util

import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexLiteral
import org.json4s.JValue
import org.json4s.JsonDSL._

import scala.collection.JavaConverters._

class PelagoValues(cluster: RelOptCluster,
                   rowType: RelDataType,
                   tuples: ImmutableList[ImmutableList[RexLiteral]],
                   traits: RelTraitSet)
  extends Values(cluster, rowType, tuples, traits) with PelagoRel {

  override def copy(traitSet: RelTraitSet, inputs: util.List[RelNode]): PelagoValues = {
    PelagoValues.create(getCluster, getRowType, getTuples)
  }

  override def implement(target: RelDeviceType, alias: String): (Binding, JValue) = {
    val op = ("operator", "values")
    val pelagoTable = PelagoTable.create(alias, getRowType)
    val vals = getTuples.asScala.map(
      t => t.asScala.map(f => emitExpression(f, List(), this))
    )

    val json: JValue = op ~ ("values", vals)

    val binding = Binding(pelagoTable, getRowType.getFieldList.asScala.toList)
    val ret: (Binding, JValue) = (binding, json)
    ret
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = super.computeSelfCost(planner, mq)
}

object PelagoValues {
  def create(cluster: RelOptCluster,
             rowType: RelDataType,
             tuples: ImmutableList[ImmutableList[RexLiteral]]): PelagoValues = {
    val traitSet = cluster.traitSet.replace(PelagoRel.CONVENTION)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.X86_64)
      .replaceIf(RelHomDistributionTraitDef.INSTANCE, () => RelHomDistribution.SINGLE)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
    assert(traitSet.containsIfApplicable(RelPacking.UnPckd))
    new PelagoValues(cluster,
      rowType,
      tuples,
      traitSet)
  }
}
