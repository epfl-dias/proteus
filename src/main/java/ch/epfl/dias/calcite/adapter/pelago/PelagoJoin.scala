package ch.epfl.dias.calcite.adapter.pelago

import java.sql.SQLType

import ch.epfl.dias.emitter.{Binding, PlanConversionException}
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.CorrelationId
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.JoinRelType
import org.apache.calcite.rel.metadata.{DefaultRelMetadataProvider, RelMdDistribution, RelMdParallelism, RelMetadataQuery}
import org.apache.calcite.rex.{RexCall, RexFieldAccess, RexInputRef, RexNode}
import org.apache.calcite.util.{ImmutableIntList, Util}
import org.json4s.{JValue, JsonAST}
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.Tuple2
import java.util

import org.apache.calcite.linq4j.tree.Primitive
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`type`.SqlTypeName

import scala.collection.mutable.ListBuffer

//import ch.epfl.dias.calcite.adapter.pelago.`trait`.RelDeviceType
import com.google.common.base.Supplier

class PelagoJoin private (cluster: RelOptCluster, traitSet: RelTraitSet, left: RelNode, right: RelNode, condition: RexNode, variablesSet: util.Set[CorrelationId], joinType: JoinRelType)
//        assert getConvention() == left.getConvention();
//        assert getConvention() == right.getConvention();
//        assert !condition.isAlwaysTrue();
  extends Join(cluster, traitSet, left, right, condition, variablesSet, joinType) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, conditionExpr: RexNode, left: RelNode, right: RelNode, joinType: JoinRelType, semiJoinDone: Boolean) = {
    PelagoJoin.create(left, right, conditionExpr, getVariablesSet, joinType)
  }

  override def estimateRowCount(mq: RelMetadataQuery): Double = super.estimateRowCount(mq)

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = { // Pelago does not support cross products
//    if (condition.isAlwaysTrue) return planner.getCostFactory.makeInfiniteCost

//    if (traitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX))) return planner.getCostFactory.makeTinyCost

//    if (getLeft.getRowType.getFieldCount > 1) return planner.getCostFactory.makeHugeCost
//    if (traitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX))) return planner.getCostFactory.makeTinyCost
//    var devFactor = if (traitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == RelDeviceType.NVPTX) 0.1 else 1

    var rowCount = mq.getRowCount(this)
    // Joins can be flipped, and for many algorithms, both versions are viable
    // and have the same cost. To make the results stable between versions of
    // the planner, make one of the versions slightly more expensive.
    //        switch (joinType) {
    //            case RIGHT:
    //                rowCount = addEpsilon(rowCount);
    //                break;
    //            default:
    //                if (RelNodes.COMPARATOR.compare(left, right) > 0) {
    //                    rowCount = addEpsilon(rowCount);
    //                }
    //        }
    // Cheaper if the smaller number of rows is coming from the LHS.
    // Model this by adding L log L to the cost.]
    val rightRowCount = right.estimateRowCount(mq)
    val leftRowCount = left.estimateRowCount(mq)

    if (leftRowCount.isInfinite) rowCount = leftRowCount
    else rowCount += Util.nLogN(leftRowCount * left.getRowType.getFieldCount)

    rowCount *= left.getRowType.getFieldCount

    if (rightRowCount.isInfinite) {
      rowCount = rightRowCount
    } else {
      rowCount += rightRowCount //For the current HJ implementation, extra fields in the probing rel are 0-cost // * 0.1 * right.getRowType().getFieldCount();
      //TODO: Cost should change for radix-HJ
    }
    rowCount *= right.getRowType.getFieldCount
    planner.getCostFactory.makeCost(rowCount*10, rowCount, 0).multiplyBy(0.1)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("build", left.getRowType.toString).item("lcount", Util.nLogN(left.estimateRowCount(left.getCluster.getMetadataQuery) * left.getRowType.getFieldCount)).item("rcount", right.estimateRowCount(right.getCluster.getMetadataQuery)).item("buildcountrow", left.estimateRowCount(left.getCluster.getMetadataQuery)).item("probecountrow", right.estimateRowCount(right.getCluster.getMetadataQuery))

//  override def estimateRowCount(mq: RelMetadataQuery): Double = mq.getRowCount(getRight) * mq.getPercentageOriginalRows(getLeft);//Math.max(mq.getRowCount(getLeft), mq.getRowCount(getRight))

  def getTypeSize(t: RelDataType) = t.getSqlTypeName match {
    case SqlTypeName.INTEGER => 32
    case SqlTypeName.BIGINT  => 64
    case SqlTypeName.BOOLEAN => 1  //TODO: check this
    case SqlTypeName.VARCHAR => 32
    case _ => throw new PlanConversionException("Unsupported type: " + t)
  }

  override def implement: (Binding, JsonAST.JValue) = {
    val op = ("operator" , "hashjoin-chained")
    val build = getLeft.asInstanceOf[PelagoRel].implement()
    val build_binding: Binding = build._1
    val build_child = build._2
    val probe = getRight.asInstanceOf[PelagoRel].implement()
    val probe_binding: Binding = probe._1
    val probe_child = probe._2

    assert(getCondition.isA(SqlKind.EQUALS), "Only equality hash joins supported")
    val joinCondOperands = getCondition.asInstanceOf[RexCall].operands
    //TODO: while the executor supports it, we should update the translator
    assert(joinCondOperands.size() == 2, "Complex equi-join (executor supports it, but translator does not)")

//    val cond = emitExpression(getCondition, List(leftBinding,rightBinding))
    val probe_k: JObject = emitExpression(joinCondOperands.get(0), List(build_binding, probe_binding)).asInstanceOf[JsonAST.JObject]
    val build_k: JObject = emitExpression(joinCondOperands.get(1), List(build_binding, probe_binding)).asInstanceOf[JsonAST.JObject]
    val alias   = "join" + getId
    val rowType = emitSchema(alias, getRowType)
//    ("attrName", joinCondOperands.get(1).asInstanceOf[RexInputRef].getName) ~ ("relName", alias)


    var build_keyRexInputRef = joinCondOperands.get(1).asInstanceOf[RexInputRef]
    var build_w = ListBuffer(32 + getTypeSize(build_keyRexInputRef.getType))

    var build_keyName = build_keyRexInputRef.getName
    val build_e = getRowType.getFieldList.asScala.zipWithIndex.flatMap {
      f => {
        if (f._2 != build_keyRexInputRef.getIndex && f._2 < getLeft.getRowType.getFieldCount) {
          build_w += getTypeSize(f._1.getType)
          List(
            emitExpression(RexInputRef.of(f._2, getRowType), List(build_binding, probe_binding))
              .asInstanceOf[JsonAST.JObject] ~
            ("register_as", ("attrName", f._1.getName) ~ ("relName", alias))
          )
//          List(("relName", alias) ~ ("attrName", f._1.getName) ~ ("type", f._1.getType.toString))
        } else {
          if (f._2 <  getRight.getRowType.getFieldCount) build_keyName = f._1.getName
          List()
        }
      }
    }.zipWithIndex.map{e => ("e", e._1) ~ ("packet", e._2 + 1) ~ ("offset", 0)}


    var probe_keyRexInputRef = joinCondOperands.get(0).asInstanceOf[RexInputRef]
    var probe_w = ListBuffer(32 + getTypeSize(probe_keyRexInputRef.getType))

    var probe_keyName = probe_keyRexInputRef.asInstanceOf[RexInputRef].getName
    val probe_e = getRowType.getFieldList.asScala.zipWithIndex.flatMap {
      f => {
        if (f._2 != probe_keyRexInputRef.asInstanceOf[RexInputRef].getIndex && f._2 >= getRight.getRowType.getFieldCount) {
          probe_w += getTypeSize(f._1.getType)
          List(
            emitExpression(RexInputRef.of(f._2, getRowType), List(build_binding, probe_binding))
              .asInstanceOf[JsonAST.JObject] ~
              ("register_as", ("attrName", f._1.getName) ~ ("relName", alias))
          )
//          List(("relName", alias) ~ ("attrName", f._1.getName) ~ ("type", f._1.getType.toString))
        } else {
          if (f._2 >= getRight.getRowType.getFieldCount) probe_keyName = f._1.getName
          List()
        }
      }
    }.zipWithIndex.map{e => ("e", e._1) ~ ("packet", e._2 + 1) ~ ("offset", 0)}



    val json = op ~
//      ("tupleType"    , rowType     ) ~
      ("gpu"              , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)                      ) ~
      ("build_k"          , build_k ~ ("register_as", ("attrName", build_keyName) ~ ("relName", alias))) ~
      ("build_e"          , build_e     ) ~
      ("build_w"          , build_w     ) ~
      ("build_input"      , build_child ) ~
      ("probe_k"          , probe_k ~ ("register_as", ("attrName", probe_keyName) ~ ("relName", alias))) ~
      ("probe_e"          , probe_e     ) ~
      ("probe_w"          , probe_w     ) ~
      ("hash_bits"        ,           Math.ceil(2 * Math.log(getLeft.estimateRowCount(getCluster.getMetadataQuery)) / Math.log(2)).asInstanceOf[Int])                 ~
      ("maxBuildInputSize", Math.min (Math.ceil(    Math.log(getCluster.getMetadataQuery.getMaxRowCount(getLeft  )) / Math.log(2)).asInstanceOf[Int], 129*1024*1024)) ~
      ("probe_input"      , probe_child )
    val binding: Binding = Binding(alias,build_binding.fields ++ probe_binding.fields)
    val ret: (Binding, JValue) = (binding,json)
    ret
  }
}


object PelagoJoin {
  def create(left: RelNode, right: RelNode, condition: RexNode, variablesSet: util.Set[CorrelationId], joinType: JoinRelType) = {
    val cluster = right.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = right.getTraitSet.replace(PelagoRel.CONVENTION)
    new PelagoJoin(cluster, traitSet, left, right, condition, variablesSet, joinType)
  }
}