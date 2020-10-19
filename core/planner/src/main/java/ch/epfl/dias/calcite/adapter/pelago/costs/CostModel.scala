package ch.epfl.dias.calcite.adapter.pelago.costs

import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.repl.Repl
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.sql.SqlKind

import scala.collection.JavaConverters._

object CostModel {
  // FIXME: get actual blockSize
  val blockSize: Double = 1024 * 1024

  def getDevCount(traitSet: RelTraitSet): Double =
    if (traitSet.containsIfApplicable(RelHomDistribution.RANDOM) || traitSet.containsIfApplicable(RelHomDistribution.BRDCST)) {
      if (traitSet.containsIfApplicable(RelDeviceType.NVPTX)) {
        Repl.gpudop
      } else {
        Repl.gpudop // FIXME: change Router to have the compute trait set to the target processor
      }
    } else {
      1
    }

//  private val uns = new mutable.HashMap[PelagoRel, mutable.Set[Long]]()

  def getNonCumulativeCost(rel: PelagoRel): Cost = rel match {
    // Converters
    case unpack: PelagoUnpack =>
      // Scan cost
      // FIXME: assuming all fields are 4 bytes in size
      MemBW(unpack.getRowType.getFieldCount * 16)
    case pack: PelagoPack =>
      // Materialization cost
      MemBW(pack.getRowType.getFieldCount * blockSize * 1024)
    case _: PelagoUnion =>
//      var cnt = 0
//      val splits = new mutable.HashSet[Long]
//      new RelVisitor() {
//        override def visit(node: RelNode, ordinal: Int, parent: RelNode): Unit = {
//          node match {
//            case s: RelSubset =>
//              if (s.getBest != null) visit(s.getBest, ordinal, parent)
//              return
//            case split: PelagoSplit =>
//              cnt += 1
//              val id: Long = split.splitId
//              if (!splits.remove(id)) splits.add(id)
//            case _ =>
//          }
//          node.childrenAccept(this)
//        }
//      }.go(rel)
//      if (cnt > 0) println(rel.getId + ": " + splits + " (" + uns.getOrElse(rel, "") + ")")
//      uns.put(rel, splits)
//      if (splits.nonEmpty) return ReallyInfiniteCost()
      MemBW(rel.getRowType.getFieldCount * blockSize * 10 /* 10 for sync cost */ ) +
        InterconnectBW(rel.getRowType.getFieldCount * 8)// * rel.getCluster.getMetadataQuery.getRowCount(rel.getInput(0)))
    case router: PelagoSplit =>
      MemBW(rel.getRowType.getFieldCount * 10 /* 10 for sync cost */ )
    case router: PelagoRouter if router.getHomDistribution == RelHomDistribution.BRDCST && router.getTraitSet.containsIfApplicable(RelPacking.UnPckd)  =>
      InfiniteCost()
    case router: PelagoRouter =>
      MemBW(router.getRowType.getFieldCount /* 10 for sync cost */ ) + // FIXME: or 100?
        InterconnectBW(router.getRowType.getFieldCount * 4e-2 * {
          if (!router.hasMemMove
            || router.getInput.isInstanceOf[PelagoSplit] // FIXME: Input may be a Subset, in which case we do not know what's the actual input, we need explicit mem-moves in the plan
            || router.getInput.isInstanceOf[PelagoUnion]){
            blockSize * 0.01 //FIXME: patch until we implement exclicit mem-moves in the plans
          } else if (router.getTraitSet.containsIfApplicable(RelPacking.Packed)){
            blockSize
          } else {
            blockSize * 2000 //FIXME: patch until we implement exclicit mem-moves in the plans
          }
        })
    case cross: PelagoDeviceCross if cross.getDeviceType() == RelDeviceType.NVPTX && cross.getTraitSet.containsIfApplicable(RelPacking.UnPckd) =>
      InfiniteCost()
    case cross: PelagoDeviceCross =>
      MemBW(cross.getRowType.getFieldCount * 10 /* 10 for sync cost */ ) + // FIXME: or 100?
        InterconnectBW((cross.getRowType.getFieldCount) * 8 * {
          if (!cross.hasMemMove
            || (cross.getInput.isInstanceOf[PelagoRouter]
                && cross.getInput.asInstanceOf[PelagoRouter].hasMemMove)
            || cross.getInput.isInstanceOf[PelagoSplit]
            || cross.getInput.isInstanceOf[PelagoUnion]) {
            blockSize * 0.01 //FIXME: patch until we implement exclicit mem-moves in the plans
          } else if (cross.getTraitSet.containsIfApplicable(RelPacking.Packed)){
            blockSize
          } else {
            blockSize * 0.2 //FIXME: patch until we implement exclicit mem-moves in the plans
          }
        })
    // Relational Operators
    case _: PelagoTableScan =>
      MemBW(16.0/blockSize)
    case _: PelagoDictTableScan => // FIXME: tune
      MemBW(8) + Compute(1024)
    case join: PelagoJoin => getNonCumulativeCost(join.asInstanceOf[Join])
    case _: PelagoFilter =>
      Compute(16)
    case agg: PelagoAggregate if agg.getAggCallList.asScala.exists(_.getAggregation.getKind == SqlKind.AVG) =>
      InfiniteCost()
    case red: PelagoAggregate if red.getGroupCount == 0 =>
      Compute(16) + MemBW(0.01 * (8/blockSize) * red.getRowType.getFieldCount)
    case agg: PelagoAggregate if agg.getGroupCount > 0 =>
      Compute(16) + RandomMemBW(0.1*agg.getRowType.getFieldCount + 0.01 * agg.getInput.getRowType.getFieldCount)
    case project: PelagoProject =>
      Compute(project.getRowType.getFieldCount * 16 /* naive estimation for number of instr / expr */ ) +
        MemBW(1)
    case _: PelagoSort =>
      MemBW(8)
    case _: PelagoTableModify =>
      MemBW(64)
    case _: PelagoValues =>
      Compute(16)
  }

  def getNonCumulativeCost(rel: PelagoToEnumerableConverter): Cost = MemBW(rel.getRowType.getFieldCount * 64 * 1024)

  def getNonCumulativeCost(rel: Join): Cost = {
    if (rel.getCondition.isAlwaysTrue || !rel.analyzeCondition().isEqui) return InfiniteCost()
//    if (rel.getCluster.getMetadataQuery.getRowCount(rel.getLeft) > rel.getCluster.getMetadataQuery.getRowCount(rel.getRight)){
//      InfiniteCost()
//    } else {
      val buildTupleBytes = (rel.getLeft.getRowType.getFieldCount * 8 + 2) * Math.log(rel.getCluster.getMetadataQuery.getRowCount(rel.getLeft))
      RandomMemBW(buildTupleBytes * rel.getRight.getRowType.getFieldCount)
      //        RandomMemBW(800 * 1024 * 1024 * join.getCluster.getMetadataQuery.getPercentageOriginalRows(join.getLeft))
//    }
  }

  def getNonCumulativeCost(rel: RelNode): Cost = rel match {
    case e: PelagoToEnumerableConverter => getNonCumulativeCost(e)
    case e: PelagoRel => getNonCumulativeCost(e)
    case e: PelagoLogicalJoin => getNonCumulativeCost(e)
    case _ => null
  }

}
