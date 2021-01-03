package ch.epfl.dias.calcite.adapter.pelago.traits

import org.apache.calcite.plan.volcano.{StealSetId, VolcanoPlanner}
import org.apache.calcite.plan.{RelOptPlanner, RelTrait}
import org.apache.calcite.rel.RelNode

import java.util

/**
  * TODO: should we convert it into a RelMultipleTrait ? Does a RelMultipleTrait has *ANY* of the values or all?
  */
object RelSplitPoint {
  private val map = new util.HashMap[Set[Long], RelSplitPoint]
  val NONE: RelSplitPoint = RelSplitPoint.of(-1)
  val TEST: RelSplitPoint = RelSplitPoint.of(100000)

  def of(point: Long): RelSplitPoint = {
    of(Set(point))
  }

  def of(point: Set[Long]): RelSplitPoint = {
    var s = map.get(point)
    if (s != null) return s
    s = new RelSplitPoint(point)
    map.put(point, s)
    s
  }

  var split_cnt: Long = 0;
  val mem = new collection.mutable.WeakHashMap[Any, Long]

  def getOrCreateId(input: RelNode): Option[Long] = {
    val set = input.getCluster.getPlanner.asInstanceOf[VolcanoPlanner].getSet(input)
    if (set == null) {
      println(input)
      println(input.getInputs.get(0))
      return Option.empty;
    }
    val id = if (set == null) input else StealSetId.getSetId(set)
    Option(mem.getOrElseUpdate(id, {
      println(split_cnt + ": " + input)
      val splitId = split_cnt
      split_cnt += 1
      splitId
    }))
  }

  def merge(s1: RelSplitPoint, s2: RelSplitPoint): RelSplitPoint = {
    if (s1 == RelSplitPoint.NONE) return s2
    if (s2 == RelSplitPoint.NONE) return s1
    if (s1 == s2) return s1
    of(s1.point ++ s2.point)
  }

  def of(r: RelNode): RelSplitPoint = of(getOrCreateId(r).get)
}

class RelSplitPoint private(val point: Set[Long]) extends PelagoTrait {
  override def toString: String = {
    if (this == RelSplitPoint.NONE) return "NoSplit"
    "Split" + point
  }

  override def getTraitDef: RelSplitPointTraitDef = RelSplitPointTraitDef.INSTANCE

  override def satisfies(t: RelTrait): Boolean = t.isInstanceOf[RelSplitPoint] && (t.asInstanceOf[RelSplitPoint].point == point)

  override def register(planner: RelOptPlanner): Unit = {
  }
}