package ch.epfl.dias.calcite.adapter.pelago.rel

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMdDeviceType
import ch.epfl.dias.calcite.adapter.pelago._
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable
import ch.epfl.dias.calcite.adapter.pelago.traits.{RelComputeDevice, RelComputeDeviceTraitDef, RelDeviceType, RelDeviceTypeTraitDef}
import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema, getFields}
import org.apache.calcite.plan.{RelOptCluster, RelOptCost, RelOptPlanner, RelTraitSet}
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.{RexInputRef, RexNode}
import org.apache.calcite.util.Util
import org.json4s.JsonDSL._
import org.json4s.{JsonAST, _}

import scala.collection.JavaConverters._

/**
  * Implementation of {@link org.apache.calcite.rel.core.Sort}
  * relational expression in Pelago.
  */
class PelagoSort protected (cluster: RelOptCluster, traits: RelTraitSet, child: RelNode, collation: RelCollation, offset: RexNode, fetch: RexNode) //        assert getConvention() == input.getConvention();
  extends Sort(cluster, traits, child, collation, offset, fetch) with PelagoRel {
  //  assert(getConvention eq PelagoRel.CONVENTION())

  override def copy(traitSet: RelTraitSet, input: RelNode, collation: RelCollation, offset: RexNode, fetch: RexNode): PelagoSort = {
    PelagoSort.create(input, collation, offset, fetch)
  }


  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    mq.getNonCumulativeCost(this)
  }

  override def computeBaseSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    val rowCount = mq.getRowCount(this)
    val bytesPerRow = getRowType.getFieldCount * 4
    val cpu = Util.nLogN(Math.max(rowCount, 1024)) * bytesPerRow * 1e20
    return planner.getCostFactory.makeCost(rowCount, cpu, 0)
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  override def implement(target: RelDeviceType, alias2: String): (Binding, JsonAST.JValue) = {
    return implementSort
    val alias = PelagoTable.create(alias2, getRowType)
    val op = ("operator", "project")
    val rowType = emitSchema(alias, getRowType)
    val child = implementUnpack
    val childBinding: Binding = child._1
    val childOp = child._2

    val projs = getRowType.getFieldList.asScala.map {
      f => {
        ("e",
          ("e",
            ("attributes" , List(("attrName", "__sorted") ~ ("relName", "__sort" + getId))) ~
            ("expression" , "argument"                                                    ) ~
            ("type"       , ("type", "record") ~ ("relName", "__sort" + getId)            ) ~
            ("argNo"      , 1                                                             )
          ) ~
          ("expression", "recordProjection"                                             ) ~
          ("attribute" , ("attrName", "__sorted") ~ ("relName", "__sort" + getId)       )
        ) ~
        ("expression" , "recordProjection") ~
        ("attribute"  , ("attrName", f.getName) ~ ("relName", "__sort" + getId)) ~
        ("register_as",
          ("attrName", f.getName) ~
          ("relName", alias.getPelagoRelName)
        )
      }
    }

    val json = op ~
      ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("e"          , projs                                                 ) ~
      ("relName"    , alias.getPelagoRelName                                ) ~
      ("input"      , childOp                                               )
    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding, json)
    ret
  }

  def implementUnpack: (Binding, JsonAST.JValue) = {
    val op = ("operator", "unpack")
    val alias = PelagoTable.create("__sort_unpack" + getId, getRowType)
    val child = implementSort
    val childOp = child._2

    val json = op ~
      ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("projections",
        List(
          ("e",
            ("attributes" , List(("attrName", "__sorted") ~ ("relName", "__sort" + getId))) ~
            ("expression" , "argument"                                                    ) ~
            ("type"       , ("type", "record") ~ ("relName", "__sort" + getId)            ) ~
            ("argNo"      , 1                                                             )
          ) ~
          ("expression", "recordProjection"                                             ) ~
          ("attribute" , ("attrName", "__sorted") ~ ("relName", "__sort" + getId)       )
        )
      ) ~
      ("input"      , childOp                                               )
    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding, json)
    ret
  }

  def implementSort: (Binding, JsonAST.JValue) = {
    val op = ("operator", "sort")
    val alias = PelagoTable.create("__sort" + getId, getRowType)
    val rowType = emitSchema(alias, getRowType)
    val child = getInput.asInstanceOf[PelagoRel].implement(getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE))
    val childBinding: Binding = child._1
    val childOp = child._2

    val colList = getCollation.getFieldCollations.asScala.map { f => f.getFieldIndex }
    val exprKey = getCollation.getFieldCollations.asScala.map {
      f => {
        ("direction", f.direction.shortString) ~
        ("expression",
          emitExpression(RexInputRef.of(f.getFieldIndex, getInput.getRowType), List(childBinding), this).asInstanceOf[JObject] ~
          ("register_as",
            ("attrName", getRowType.getFieldNames.get(f.getFieldIndex)) ~ ("relName", alias.getPelagoRelName)
          )
        )
      }
    }.toList
    val exprVal = getRowType.getFieldList.asScala.flatMap {
      f => {
        if (colList.contains(f.getIndex)){
          List()
        } else {
          List(
            ("direction", "NONE") ~
            ("expression",
              emitExpression(RexInputRef.of(f.getIndex, getInput.getRowType), List(childBinding), this).asInstanceOf[JObject] ~
              ("register_as",
                ("attrName", f.getName) ~ ("relName", alias.getPelagoRelName)
              )
            )
          )
        }
      }
    }.toList

    val json = op ~
      ("gpu"        , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("rowType"    , rowType                                               ) ~
      ("e"          , exprKey ++ exprVal                                    ) ~
      ("granularity", "thread"                                              ) ~
      ("input"      , childOp                                               )
    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding, json)
    ret
  }
}

object PelagoSort{
  def create(input: RelNode, collation: RelCollation, offset: RexNode, fetch: RexNode): PelagoSort = {
    val cluster  = input.getCluster
    val mq       = cluster.getMetadataQuery
    val traitSet = input.getTraitSet.replace(PelagoRel.CONVENTION)
      .replace(RelCollationTraitDef.INSTANCE.canonize(collation))
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () => PelagoRelMdDeviceType.sort(mq, input))
      .replaceIf(RelComputeDeviceTraitDef.INSTANCE, () => RelComputeDevice.from(input))
    new PelagoSort(cluster, traitSet, input, collation, offset, fetch)
  }
}