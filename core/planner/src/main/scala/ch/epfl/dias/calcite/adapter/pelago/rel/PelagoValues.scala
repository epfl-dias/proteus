package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelComputeDevice, RelComputeDeviceTraitDef, RelDeviceType, RelDeviceTypeTraitDef, RelHomDistribution, RelHomDistributionTraitDef, RelPacking, RelPackingTraitDef}
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON._
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexLiteral
import org.json4s.JsonDSL._
import org.json4s._

import java.util
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

    val vals: JValue = getRowType.getFieldList.asScala.map(
      f => {
        ("type", emitType(f.getType, List())) ~
        ("attrName", f.getName) ~
        ("v",
          getTuples.asScala.map(t => emitLiteral(t.get(f.getIndex)))
        )
      }
    )

    val json: JValue = op ~ ("values", vals) ~ ("relName", pelagoTable.getPelagoRelName)

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
      .replaceIf(RelPackingTraitDef.INSTANCE, () => RelPacking.Packed)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.X86_64)
      .replaceIf(RelHomDistributionTraitDef.INSTANCE, () => RelHomDistribution.SINGLE)
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => RelDeviceType.X86_64)
    assert(traitSet.containsIfApplicable(RelPacking.Packed))
    new PelagoValues(cluster,
      rowType,
      tuples,
      traitSet)
  }
}
