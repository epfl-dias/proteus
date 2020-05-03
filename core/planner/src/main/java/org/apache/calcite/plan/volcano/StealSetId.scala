package org.apache.calcite.plan.volcano

object StealSetId {
  def getSetId(relSet: RelSet): Int = relSet.id
}
