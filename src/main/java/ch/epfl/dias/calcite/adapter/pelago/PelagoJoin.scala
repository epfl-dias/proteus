package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.emitter.{Binding, PlanConversionException}
import ch.epfl.dias.emitter.PlanToJSON._
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel._
import org.apache.calcite.rel.core._
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex._
import org.apache.calcite.util.{ImmutableIntList, Util}
import org.json4s.{JValue, JsonAST}
import org.json4s.JsonDSL._
import org.json4s._

import scala.collection.JavaConverters._
import java.util

import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`type`.SqlTypeName

import scala.collection.mutable.ListBuffer

class PelagoJoin private (cluster: RelOptCluster, traitSet: RelTraitSet, left: RelNode, right: RelNode,
                          condition: RexNode, leftKeys: ImmutableIntList, rightKeys: ImmutableIntList,
                          variablesSet: util.Set[CorrelationId], joinType: JoinRelType)
//        assert getConvention() == left.getConvention();
//        assert getConvention() == right.getConvention();
//        assert !condition.isAlwaysTrue();
  extends EquiJoin(cluster, traitSet, left, right, condition, leftKeys, rightKeys, variablesSet, joinType) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, conditionExpr: RexNode, left: RelNode, right: RelNode, joinType: JoinRelType,
                    semiJoinDone: Boolean) = {
    PelagoJoin.create(left, right, conditionExpr, getVariablesSet, joinType)
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = { // Pelago does not support cross products
    if (condition.isAlwaysTrue) return planner.getCostFactory.makeHugeCost
    //    if (!getCondition.isA(SqlKind.EQUALS)) return planner.getCostFactory.makeHugeCost

    val rf = {
      if (!getTraitSet.containsIfApplicable(RelDistributions.SINGLETON)) {
        if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) 0.000001
        else 100//0.1
      } else if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
        10000.0 //0.01
      } else {
        1000000.0
      }
    }

    //    if (getLeft.getRowType.getFieldCount > 1) return planner.getCostFactory.makeHugeCost
    //    if (traitSet.satisfies(RelTraitSet.createEmpty().plus(RelDeviceType.NVPTX))) return planner.getCostFactory.makeTinyCost
    //    var devFactor = if (traitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == RelDeviceType.NVPTX) 0.1 else 1

    val rowCount = mq.getRowCount(this)
    var rc1 = rowCount
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

    if (leftRowCount.isInfinite) rc1 = leftRowCount
    else rc1 += Util.nLogN(leftRowCount * left.getRowType.getFieldCount)

    rc1 *= left.getRowType.getFieldCount * 10000

    val rc2 = if (rightRowCount.isInfinite) {
      rightRowCount
    } else {
      rightRowCount //For the current HJ implementation, extra fields in the probing rel are 0-cost // * 0.1 * right.getRowType().getFieldCount();
      //TODO: Cost should change for radix-HJ
    }
    rc1 += rc2 * right.getRowType.getFieldCount
    planner.getCostFactory.makeCost(rowCount, rc1 * 10000 * rf, 0)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString).item("build", left.getRowType.toString).item("lcount", Util.nLogN(left.estimateRowCount(left.getCluster.getMetadataQuery) * left.getRowType.getFieldCount)).item("rcount", right.estimateRowCount(right.getCluster.getMetadataQuery)).item("buildcountrow", left.estimateRowCount(left.getCluster.getMetadataQuery)).item("probecountrow", right.estimateRowCount(right.getCluster.getMetadataQuery))

//  override def estimateRowCount(mq: RelMetadataQuery): Double = mq.getRowCount(getRight) * mq.getPercentageOriginalRows(getLeft);//Math.max(mq.getRowCount(getLeft), mq.getRowCount(getRight))

  def getTypeSize(t: RelDataType) = t.getSqlTypeName match {
    case SqlTypeName.INTEGER => 32
    case SqlTypeName.BIGINT  => 64
    case SqlTypeName.BOOLEAN => 1  //TODO: check this
    case SqlTypeName.VARCHAR => 32
    case SqlTypeName.DOUBLE  => 64
    case SqlTypeName.DATE    => 64
    case _ => throw new PlanConversionException("Unsupported type: " + t)
  }

  def getCompositeKeyExpr(operands: ImmutableList[RexNode], arg: Int, bindings: List[Binding], alias: String):  (JObject, RexInputRef, Int)={
    var size = 0
    (
      ("expression", "recordConstruction") ~ ("type", ("type", "record")) ~ ("attributes",
        operands.asScala.zipWithIndex.map(p => {
          assert(p._1.isA(SqlKind.EQUALS), "Only equality hash joins supported, found: " + getCondition)
          var op = p._1.asInstanceOf[RexCall].operands.get(arg)
          val op_other = p._1.asInstanceOf[RexCall].operands.get(1 - arg)
          if (op.isInstanceOf[RexInputRef] && op_other.isInstanceOf[RexInputRef]){
            val rt = op.asInstanceOf[RexInputRef].getIndex
            val ro = op_other.asInstanceOf[RexInputRef].getIndex
            if ((arg == 0) != (rt < ro)) op = op_other
          }
          size += getTypeSize(op.getType)
          val name = if (op.isInstanceOf[RexInputRef]) op.asInstanceOf[RexInputRef].getName else p._2.toString
          val je = emitExpression(op, bindings).asInstanceOf[JsonAST.JObject]
          (
            "name",
            name
          ) ~ ("e", je ~ ("register_as", ("attrName", name) ~ ("relName", alias)))
        }).toList
      ),
      new RexInputRef(bindings.map(e => e.fields.length).sum + arg, getCluster.getTypeFactory.createSqlType(SqlTypeName.INTEGER)),
      size
    )
  }

  def getKeyExpr(cond: RexNode, arg: Int, bindings: List[Binding], alias: String): (JObject, RexInputRef, Int) = {
    assert(cond.isA(SqlKind.EQUALS) || cond.isA(SqlKind.AND), "Only equality hash joins supported, found: " + cond)
    val operands = cond.asInstanceOf[RexCall].operands

    if (cond.isA(SqlKind.EQUALS)){
      val inp = operands.get(arg).asInstanceOf[RexInputRef]
      val inp_other = operands.get(1 - arg).asInstanceOf[RexInputRef]
      val rt = inp.getIndex
      val ro = inp_other.getIndex
      (
        emitExpression(operands.get(if ((arg == 0) == (rt < ro)) arg else (1 - arg)), bindings).asInstanceOf[JObject],
        inp,
        getTypeSize(inp.getType)
      )
    } else {
      assert(cond.isA(SqlKind.AND), "Only equality hash joins supported, found: " + cond)
      getCompositeKeyExpr(operands, arg, bindings, alias)
    }
  }

  override def implement(target: RelDeviceType): (Binding, JValue) = {
    val op = ("operator" , "hashjoin-chained")
    val build = getLeft.asInstanceOf[PelagoRel].implement(target)
    val build_binding: Binding = build._1
    val build_child = build._2
    val probe = getRight.asInstanceOf[PelagoRel].implement(target)
    val probe_binding: Binding = probe._1
    val probe_child = probe._2

    val alias   = "join" + getId

    //FIXME: joinCondOperands does not always belong to the probe side
//    val cond = emitExpression(getCondition, List(leftBinding,rightBinding))
    val (probe_k, probe_keyRexInputRef, pk_size) = getKeyExpr(getCondition, 1, List(build_binding, probe_binding), alias)
    val (build_k, build_keyRexInputRef, bk_size) = getKeyExpr(getCondition, 0, List(build_binding, probe_binding), alias)
    val rowType = emitSchema(alias, getRowType)
//    ("attrName", joinCondOperands.get(1).asInstanceOf[RexInputRef].getName) ~ ("relName", alias)


    val joinCondOperands = getCondition.asInstanceOf[RexCall].operands
    var build_w = ListBuffer(32 + bk_size)

    var build_keyName  = build_keyRexInputRef.getName
    var build_keyIndex = build_keyRexInputRef.getIndex
    val build_e = getRowType.getFieldList.asScala.zipWithIndex.flatMap {
      f => {
        if (f._2 != build_keyIndex && f._2 < getLeft.getRowType.getFieldCount) {
          build_w += getTypeSize(f._1.getType)
          List(
            emitExpression(RexInputRef.of(f._2, getRowType), List(build_binding, probe_binding))
              .asInstanceOf[JsonAST.JObject] ~
            ("register_as", ("attrName", f._1.getName) ~ ("relName", alias))
          )
//          List(("relName", alias) ~ ("attrName", f._1.getName) ~ ("type", f._1.getType.toString))
        } else {
          if (f._2 <  getLeft.getRowType.getFieldCount) build_keyName = f._1.getName
          List()
        }
      }
    }.zipWithIndex.map{e => ("e", e._1) ~ ("packet", e._2 + 1) ~ ("offset", 0)} //FIXME: using different packets for each of them is the worst performance-wise

    var probe_w = ListBuffer(32 + pk_size)

    var probe_keyName = probe_keyRexInputRef.asInstanceOf[RexInputRef].getName
    val probe_e = getRowType.getFieldList.asScala.zipWithIndex.flatMap {
      f => {
        if (f._2 != probe_keyRexInputRef.asInstanceOf[RexInputRef].getIndex && f._2 >= getLeft.getRowType.getFieldCount) {
          probe_w += getTypeSize(f._1.getType)
          List(
            emitExpression(RexInputRef.of(f._2, getRowType), List(build_binding, probe_binding))
              .asInstanceOf[JsonAST.JObject] ~
              ("register_as", ("attrName", f._1.getName) ~ ("relName", alias))
          )
//          List(("relName", alias) ~ ("attrName", f._1.getName) ~ ("type", f._1.getType.toString))
        } else {
          if (f._2 >= getLeft.getRowType.getFieldCount) probe_keyName = f._1.getName
          List()
        }
      }
    }.zipWithIndex.map{e => ("e", e._1) ~ ("packet", e._2 + 1) ~ ("offset", 0)} //FIXME: using different packets for each of them is the worst performance-wise

    val rowEst = Math.min(getLeft.estimateRowCount(getCluster.getMetadataQuery), 512*1024*1024)
    val maxrow = getCluster.getMetadataQuery.getMaxRowCount(getLeft  )
    val maxEst = if (maxrow != null) Math.min(maxrow, 64*1024*1024) else 64*1024*1024

    val hash_bits = Math.min(1 + Math.ceil(Math.log(rowEst)/Math.log(2)).asInstanceOf[Int], 20)

    val json = op ~
//      ("tupleType"        , rowType     ) ~
      ("gpu"              , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX)                       ) ~
      ("build_k"          , build_k ~ ("register_as", ("attrName", build_keyName) ~ ("relName", alias)) ) ~
      ("build_e"          , build_e                                                                     ) ~
      ("build_w"          , build_w                                                                     ) ~
      ("build_input"      , build_child                                                                 ) ~
      ("probe_k"          , probe_k ~ ("register_as", ("attrName", probe_keyName) ~ ("relName", alias)) ) ~
      ("probe_e"          , probe_e                                                                     ) ~
      ("probe_w"          , probe_w                                                                     ) ~
      ("hash_bits"        , hash_bits                                                                   ) ~
      ("maxBuildInputSize", maxEst.asInstanceOf[Int]                                                    ) ~
      ("probe_input"      , probe_child                                                                 )
    val binding: Binding = Binding(alias,build_binding.fields ++ probe_binding.fields)
    val ret: (Binding, JValue) = (binding,json)
    ret
  }
}


object PelagoJoin {
  def create(left: RelNode, right: RelNode, condition: RexNode,
             variablesSet: util.Set[CorrelationId], joinType: JoinRelType) = {
    val info = JoinInfo.of(left, right, condition)
//    assert(info.isEqui, "should had been equi-join!")

    val cluster = right.getCluster
    val mq = cluster.getMetadataQuery
    val traitSet = right.getTraitSet.replace(PelagoRel.CONVENTION)

    assert(right.getTraitSet.containsIfApplicable(RelPacking.UnPckd))
    assert(left.getTraitSet.containsIfApplicable(RelPacking.UnPckd))

    new PelagoJoin(cluster, traitSet, left, right, condition, info.leftKeys, info.rightKeys, variablesSet, joinType)
  }
}