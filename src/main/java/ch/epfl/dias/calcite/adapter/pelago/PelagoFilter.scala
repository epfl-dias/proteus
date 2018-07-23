package ch.epfl.dias.calcite.adapter.pelago

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.{RelDeviceType, RelDeviceTypeTraitDef}
import ch.epfl.dias.emitter.PlanToJSON.{emitAggExpression, emitArg, emitExpression, emitSchema, emit_, getFields}
import ch.epfl.dias.emitter.Binding
import com.google.common.base.Supplier
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.{RelNode, RelWriter}
import org.apache.calcite.rel.core.{AggregateCall, Filter}
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization


class PelagoFilter protected (cluster: RelOptCluster, traitSet: RelTraitSet, input: RelNode, condition: RexNode) extends Filter(cluster, traitSet, input, condition) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, input: RelNode, condition: RexNode) = PelagoFilter.create(input, condition)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    if (getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)) super.computeSelfCost(planner, mq).multiplyBy(0.001)
    else super.computeSelfCost(planner, mq).multiplyBy(10)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("isS", getTraitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX)).toString)

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    val op = ("operator" , "select")
    val child = getInput.asInstanceOf[PelagoRel].implement(target)
    val childBinding: Binding = child._1
    val childOp = child._2
    val rowType = emitSchema(childBinding.rel, getRowType)
    val cond = emitExpression(getCondition, List(childBinding))

    val json : JValue = op ~
      ("gpu"      , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("p"        , cond                                                  ) ~
      ("input"    , childOp                                               )

    val ret: (Binding, JValue) = (childBinding, json)
    ret
  }
}

object PelagoFilter{
  def create(input: RelNode, condition: RexNode): PelagoFilter = {
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION)
    new PelagoFilter(input.getCluster, traitSet, input, condition)
  }
}