package ch.epfl.dias.calcite.adapter.pelago

import java.util

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery
import ch.epfl.dias.emitter.PlanToJSON._
import ch.epfl.dias.emitter.{Binding, PlanConversionException}
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel._
import org.apache.calcite.rel.`type`.RelDataType
import org.apache.calcite.rel.core._
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex._
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`type`.SqlTypeName
import org.apache.calcite.util.{ImmutableIntList, Util}
import org.json4s.JsonDSL._
import org.json4s.{JValue, JsonAST, _}

import scala.collection.JavaConverters._

class PelagoJoin private (cluster: RelOptCluster, traitSet: RelTraitSet, left: RelNode, right: RelNode,
                          condition: RexNode, leftKeys: ImmutableIntList, rightKeys: ImmutableIntList,
                          variablesSet: util.Set[CorrelationId], joinType: JoinRelType)
//        assert getConvention() == left.getConvention();
//        assert getConvention() == right.getConvention();
//        assert !condition.isAlwaysTrue();
  extends Join(cluster, traitSet, left, right, condition, variablesSet, joinType) with PelagoRel {
  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, conditionExpr: RexNode, left: RelNode, right: RelNode, joinType: JoinRelType,
                    semiJoinDone: Boolean) = {
    PelagoJoin.create(left, right, conditionExpr, getVariablesSet, joinType)
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    if (condition.isAlwaysTrue || !analyzeCondition.isEqui) return planner.getCostFactory.makeHugeCost
    //    if (!getCondition.isA(SqlKind.EQUALS)) return planner.getCostFactory.makeHugeCost

    val rf = {
      if (!getTraitSet.containsIfApplicable(RelHomDistribution.SINGLE)) {
        if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) 0.000001
        else 100//0.1
      } else if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
        1e10 //0.01
      } else {
        1e15
      }
    }

    val rf2 = {
      if (getTraitSet.containsIfApplicable(RelHetDistribution.SINGLETON)) {
        1e8
//        return planner.getCostFactory.makeHugeCost()
      } else {
        1e-5
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
    val rightRowCount = getCluster.getMetadataQuery.getRowCount(getRight)
    val leftRowCount = getCluster.getMetadataQuery.getRowCount(getLeft)

    val rightCols = right.getRowType.getFieldCount
    val leftCols = left.getRowType.getFieldCount

    if (leftRowCount.isInfinite) rc1 = leftRowCount
    else rc1 += Util.nLogN(leftRowCount * leftCols)

    rc1 *= leftCols

    val rc2 = if (rightRowCount.isInfinite) {
      rightRowCount
    } else {
      rightRowCount //For the current HJ implementation, extra fields in the probing rel are 0-cost // * 0.1 * right.getRowType().getFieldCount();
      //TODO: Cost should change for radix-HJ
    }
    rc1 += rc2 * rightCols * 1e-5
    planner.getCostFactory.makeCost(rowCount * rf2 * rf, rc1 * rf * rf2, 0)
  }

  protected lazy val rowEst: Long = {
    val exprow: Double = try {
      getCluster.getMetadataQuery.getRowCount(getLeft)
    } catch {
      case _: Throwable => 1e20
    }
    Math.min(exprow, 64 * 1024 * 1024)
  }.asInstanceOf[Long]

  protected lazy val maxEst: Long = {
    val maxrow = getCluster.getMetadataQuery.getMaxRowCount(getLeft)
    if (maxrow != null) Math.min(maxrow, 64 * 1024 * 1024) else 64 * 1024 * 1024
  }.asInstanceOf[Long]

  protected lazy val maxBuildInputSize: Long = {
    (Math.min(rowEst, maxEst) + maxEst)/2
  }

  protected lazy val hash_bits: Long = {
    Math.ceil(Math.log(maxBuildInputSize)/Math.log(2)) + 1
  }.asInstanceOf[Long]

  override def explainTerms(pw: RelWriter): RelWriter = {
    val mq = getCluster.getMetadataQuery
    val leftRows: Double = try {
      getCluster.getMetadataQuery.getRowCount(getLeft)
    } catch {
      case _: Throwable => 1e20
    }
    val rightRows: Double = try {
      mq.getRowCount(getRight)
    } catch {
      case _: Throwable => 1e20
    }
    val rows = Math.max(leftRows, rightRows)
    val small = if (leftRows < rightRows) getLeft else getRight
    val sel = mq.getPercentageOriginalRows(small)
    super.explainTerms(pw)
//      .item("rct", mq.getRowCount(this) * sel)
      .item("tr", getTraitSet)
      .item("sel", sel)
      .item("hash_bits", hash_bits)
      .item("maxBuildInputSize", maxBuildInputSize)
      .item("buildcountrow", getCluster.getMetadataQuery.getRowCount(getLeft))
      .item("probecountrow", getCluster.getMetadataQuery.getRowCount(getRight))
      .item("ht", left.getRowType.toString)
  }

//  override def estimateRowCount(mq: RelMetadataQuery): Double = mq.getRowCount(getRight) * mq.getPercentageOriginalRows(getLeft);//Math.max(mq.getRowCount(getLeft), mq.getRowCount(getRight))

  private def getTypeSize(t: RelDataType) = t.getSqlTypeName match {
    case SqlTypeName.INTEGER    => 32
    case SqlTypeName.BIGINT     => 64
    case SqlTypeName.BOOLEAN    => 1  //TODO: check this
    case SqlTypeName.VARCHAR    => 32
    case SqlTypeName.DOUBLE     => 64
    case SqlTypeName.DATE       => 64
    case SqlTypeName.TIMESTAMP  => 64
    case _ => throw new PlanConversionException("Unsupported type: " + t)
  }

  def getCompositeKeyExpr(operands: ImmutableList[RexNode], arg: Int, bindings: List[Binding], alias: String):  (JObject, RexInputRef, Int)={
    var size = 0
    var maxsize = 0
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
          val s = getTypeSize(op.getType)
          size += s
          maxsize = Math.max(maxsize, s)
          val name = if (op.isInstanceOf[RexInputRef]) op.asInstanceOf[RexInputRef].getName else p._2.toString
          val je = emitExpression(op, bindings, this).asInstanceOf[JsonAST.JObject]
          (
            "name",
            name
          ) ~ ("e", je ~ ("register_as", ("attrName", name) ~ ("relName", alias)))
        }).toList
      ),
      new RexInputRef(bindings.map(e => e.fields.length).sum + arg, getCluster.getTypeFactory.createSqlType(SqlTypeName.INTEGER)),
      maxsize * ((size + maxsize - 1)/maxsize)
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
        emitExpression(operands.get(if ((arg == 0) == (rt < ro)) arg else (1 - arg)), bindings, this).asInstanceOf[JObject],
        inp,
        getTypeSize(inp.getType)
      )
    } else {
      assert(cond.isA(SqlKind.AND), "Only equality hash joins supported, found: " + cond)
      getCompositeKeyExpr(operands, arg, bindings, alias)
    }
  }

  override def implement(target: RelDeviceType, alias2: String): (Binding, JValue) = {
    val alias = PelagoTable.create(alias2, getRowType)
    val op = ("operator" , "hashjoin-chained")
    val build = getLeft.asInstanceOf[PelagoRel].implement(target)
    val build_binding: Binding = build._1
    val build_child = build._2
    val probe = getRight.asInstanceOf[PelagoRel].implement(target)
    val probe_binding: Binding = probe._1
    val probe_child = probe._2

    //FIXME: joinCondOperands does not always belong to the probe side
    val (probe_k, probe_keyRexInputRef, _) = getKeyExpr(getCondition, 1, List(build_binding, probe_binding), alias.getPelagoRelName)
    val (build_k, build_keyRexInputRef, _) = getKeyExpr(getCondition, 0, List(build_binding, probe_binding), alias.getPelagoRelName)

    val build_keyName  = build_keyRexInputRef.getName

    val probe_keyName = probe_keyRexInputRef.asInstanceOf[RexInputRef].getName

    val json = op ~
      ("build_k"          , build_k ~ ("register_as", ("attrName", build_keyName) ~ ("relName", alias.getPelagoRelName)) ) ~
      ("build_input"      , build_child                                                                 ) ~
      ("probe_k"          , probe_k ~ ("register_as", ("attrName", probe_keyName) ~ ("relName", alias.getPelagoRelName)) ) ~
      ("hash_bits"        , hash_bits                                                                   ) ~
      ("maxBuildInputSize", maxBuildInputSize                                                           ) ~
      ("probe_input"      , probe_child                                                                 )
    val binding: Binding = Binding(alias, getFields(getRowType))
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
    val dev = ImmutableList.of(left, right).stream().map[RelComputeDevice]((e) => mq.asInstanceOf[PelagoRelMetadataQuery].computeType(e))
    val traitSet = right.getTraitSet.replace(PelagoRel.CONVENTION)
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(ImmutableList.of(RelComputeDevice.from(dev), RelComputeDevice.from(left), RelComputeDevice.from(right)).stream()))
      .replaceIf(RelSplitPointTraitDef.INSTANCE, () => {
        RelSplitPoint.merge(left.getTraitSet.getTrait(RelSplitPointTraitDef.INSTANCE), right.getTraitSet.getTrait(RelSplitPointTraitDef.INSTANCE))
      })

    assert(right.getTraitSet.containsIfApplicable(RelPacking.UnPckd))
    assert(left.getTraitSet.containsIfApplicable(RelPacking.UnPckd))

    new PelagoJoin(cluster, traitSet, left, right, condition, info.leftKeys, info.rightKeys, variablesSet, joinType)
  }
}