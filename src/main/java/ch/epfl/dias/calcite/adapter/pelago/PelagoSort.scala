package ch.epfl.dias.calcite.adapter.pelago

import ch.epfl.dias.emitter.Binding
import ch.epfl.dias.emitter.PlanToJSON
import com.google.common.base.Supplier
import org.apache.calcite.adapter.java.JavaTypeFactory
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel._
import org.apache.calcite.rel.core.{Project, Sort}
import org.apache.calcite.rel.metadata.{RelMdCollation, RelMdDistribution, RelMetadataQuery}
import org.apache.calcite.rel
import org.apache.calcite.rel.`type`.RelDataType
import org.json4s.JsonDSL._
import org.json4s._
import org.json4s.jackson.JsonMethods._
import org.json4s.jackson.Serialization
import org.apache.calcite.rex.{RexInputRef, RexNode}
import org.json4s.JsonAST

import scala.collection.JavaConverters._
import scala.Tuple2
import java.util

import ch.epfl.dias.calcite.adapter.pelago.metadata.{PelagoRelMdDeviceType, PelagoRelMdDistribution, RelMdDeviceType}
import ch.epfl.dias.emitter.PlanToJSON.{emitExpression, emitSchema, emit_, getFields}
import org.apache.calcite.adapter.enumerable.EnumerableConvention
import org.json4s

import scala.collection.mutable

/**
  * Implementation of {@link org.apache.calcite.rel.core.Sort}
  * relational expression in Pelago.
  */
class PelagoSort protected (cluster: RelOptCluster, traits: RelTraitSet, child: RelNode, collation: RelCollation, offset: RexNode, fetch: RexNode) //        assert getConvention() == input.getConvention();
  extends Sort(cluster, traits, child, collation, offset, fetch) with PelagoRel {
  //  assert(getConvention eq PelagoRel.CONVENTION)

  override def copy(traitSet: RelTraitSet, input: RelNode, collation: RelCollation, offset: RexNode, fetch: RexNode): PelagoSort = {
    PelagoSort.create(input, collation, offset, fetch)
  }

  override def computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost = {
    if (getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) != null && getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE) == RelDeviceType.NVPTX) {
      super.computeSelfCost(planner, mq).multiplyBy(0.001)
    } else {
      super.computeSelfCost(planner, mq).multiplyBy(1000)
    }
  }

  override def explainTerms(pw: RelWriter): RelWriter = super.explainTerms(pw).item("trait", getTraitSet.toString)

  override def implement(target: RelDeviceType): (Binding, JsonAST.JValue) = {
    val op = ("operator", "project")
    val alias = "sort" + getId
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
          ("relName", alias)
        )
      }
    }

    val json = op ~
      ("gpu"    , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("e"      , projs                                                 ) ~
      ("relName", alias                                                 ) ~
      ("input"  , childOp                                               )
    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding, json)
    ret
  }

  def implementUnpack: (Binding, JsonAST.JValue) = {
    val op = ("operator", "block-to-tuples")
    val alias = "__sort_unpack" + getId
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
    val alias = "__sort" + getId
    val rowType = emitSchema(alias, getRowType)
    val child = getInput.asInstanceOf[PelagoRel].implement(getTraitSet.getTrait(RelDeviceTypeTraitDef.INSTANCE))
    val childBinding: Binding = child._1
    val childOp = child._2

    val colList = getCollation.getFieldCollations.asScala.map { f => f.getFieldIndex }
    val exprKey = getCollation.getFieldCollations.asScala.map {
      f => {
        ("direction", f.direction.shortString) ~
        ("expression",
          emitExpression(RexInputRef.of(f.getFieldIndex, getInput.getRowType), List(childBinding)).asInstanceOf[JObject] ~
          ("register_as",
            ("attrName", getRowType.getFieldNames.get(f.getFieldIndex)) ~ ("relName", alias)
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
              emitExpression(RexInputRef.of(f.getIndex, getInput.getRowType), List(childBinding)).asInstanceOf[JObject] ~
              ("register_as",
                ("attrName", f.getName) ~ ("relName", alias)
              )
            )
          )
        }
      }
    }.toList

    val json = op ~
      ("gpu"  , getTraitSet.containsIfApplicable(RelDeviceType.NVPTX) ) ~
      ("rowType", rowType) ~
      ("e"    , exprKey ++ exprVal                                    ) ~
      ("input", childOp                                               )
    val binding: Binding = Binding(alias, getFields(getRowType))
    val ret: (Binding, JValue) = (binding, json)
    ret
  }
}

object PelagoSort{
  def create(input: RelNode, collation: RelCollation, offset: RexNode, fetch: RexNode): PelagoSort = {
    val cluster  = input.getCluster
    val mq       = cluster.getMetadataQuery
    val traitSet = cluster.traitSet.replace(PelagoRel.CONVENTION)
      .replace(RelCollationTraitDef.INSTANCE.canonize(collation))
      .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier[RelDistribution]() {
        override def get: RelDistribution = {
          return RelMdDistribution.sort(mq, input)
        }
      })
      .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier[RelDeviceType]() {
        override def get: RelDeviceType = {
          return RelMdDeviceType.sort(mq, input)
        }
      });
    new PelagoSort(cluster, traitSet, input, collation, offset, fetch)
  }
}