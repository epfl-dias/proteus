/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
 */

package ch.epfl.dias.calcite.adapter.pelago.metadata

import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.rel.metadata.{
  BuiltInMetadata,
  ChainedRelMetadataProvider,
  MetadataDef,
  MetadataHandler,
  ReflectiveRelMetadataProvider,
  RelMdExpressionLineage,
  RelMetadataProvider,
  RelMetadataQuery
}
import org.apache.calcite.rel.`type`.RelDataTypeField
import org.apache.calcite.rex.RexBuilder
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexShuttle
import org.apache.calcite.rex.RexTableInputRef
import org.apache.calcite.util.BuiltInMethod
import org.apache.calcite.util.ImmutableBitSet
import com.google.common.collect.ImmutableList
import com.google.common.collect.ImmutableSet
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoDeviceCross
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoPack
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoRouter
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoTableScan
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoUnpack
import org.codehaus.commons.nullanalysis.Nullable

import scala.collection.JavaConverters._
import java.util

object PelagoRelMdExpressionLineage {
  private val INSTANCE = new PelagoRelMdExpressionLineage
  val SOURCE: RelMetadataProvider = ChainedRelMetadataProvider.of(
    ImmutableList.of(
      ReflectiveRelMetadataProvider.reflectiveSource(
        BuiltInMethod.EXPRESSION_LINEAGE.method,
        PelagoRelMdExpressionLineage.INSTANCE
      ),
      RelMdExpressionLineage.SOURCE
    )
  )

  private def createAllPossibleExpressions(
      rexBuilder: RexBuilder,
      expr: RexNode,
      predFieldsUsed: ImmutableBitSet,
      mapping: util.Map[RexInputRef, util.Set[RexNode]],
      singleMapping: util.Map[RexInputRef, RexNode]
  ): util.HashSet[RexNode] = {
    val inputRef = mapping.keySet.iterator.next
    val replacements = mapping.remove(inputRef)
    val result = new util.HashSet[RexNode]
    assert(!replacements.isEmpty)
    if (predFieldsUsed.indexOf(inputRef.getIndex) != -1) {
      for (replacement <- replacements.asScala) {
        singleMapping.put(inputRef, replacement)
        createExpressions(
          rexBuilder,
          expr,
          predFieldsUsed,
          mapping,
          singleMapping,
          result
        )
        singleMapping.remove(inputRef)
      }
    } else
      createExpressions(
        rexBuilder,
        expr,
        predFieldsUsed,
        mapping,
        singleMapping,
        result
      )
    mapping.put(inputRef, replacements)
    result
  }

  /**
    * Replaces expressions with their equivalences. Note that we only have to
    * look for RexInputRef.
    */
  private class RexReplacer private[metadata] (
      val replacementValues: util.Map[RexInputRef, RexNode]
  ) extends RexShuttle {
    override def visitInputRef(inputRef: RexInputRef): RexNode =
      replacementValues.get(inputRef)
  }
  private def extractInputRefs(expr: RexNode) = {
    val inputExtraFields = new util.LinkedHashSet[RelDataTypeField]
    val inputFinder = new RelOptUtil.InputFinder(inputExtraFields)
    expr.accept(inputFinder)
    inputFinder.build
  }
  @Nullable protected def createAllPossibleExpressions(
      rexBuilder: RexBuilder,
      expr: RexNode,
      mapping: util.Map[RexInputRef, util.Set[RexNode]]
  ): util.Set[RexNode] = { // Extract input fields referenced by expression
    val predFieldsUsed = extractInputRefs(expr)
    if (predFieldsUsed.isEmpty) { // The unique expression is the input expression
      return ImmutableSet.of(expr)
    }
    try createAllPossibleExpressions(
      rexBuilder,
      expr,
      predFieldsUsed,
      mapping,
      new util.HashMap[RexInputRef, RexNode]
    )
    catch {
      case e: UnsupportedOperationException =>
// There may be a RexNode unsupported by RexCopier, just return null
        null
    }
  }
  private def createExpressions(
      rexBuilder: RexBuilder,
      expr: RexNode,
      predFieldsUsed: ImmutableBitSet,
      mapping: util.Map[RexInputRef, util.Set[RexNode]],
      singleMapping: util.Map[RexInputRef, RexNode],
      result: util.Set[RexNode]
  ) =
    if (mapping.isEmpty) {
      val replacer = new PelagoRelMdExpressionLineage.RexReplacer(singleMapping)
      val updatedPreds = new util.ArrayList[RexNode](
        RelOptUtil.conjunctions(rexBuilder.copy(expr))
      )
      replacer.mutate(updatedPreds)
      result.addAll(updatedPreds)
    } else
      result.addAll(
        createAllPossibleExpressions(
          rexBuilder,
          expr,
          predFieldsUsed,
          mapping,
          singleMapping
        )
      )
}
class PelagoRelMdExpressionLineage
    extends MetadataHandler[BuiltInMetadata.ExpressionLineage] {
  override def getDef: MetadataDef[BuiltInMetadata.ExpressionLineage] =
    BuiltInMetadata.ExpressionLineage.DEF

  def getExpressionLineage(
      rel: PelagoPack,
      mq: RelMetadataQuery,
      outputExpression: RexNode
  ): util.Set[RexNode] = mq.getExpressionLineage(rel.getInput, outputExpression)

  def getExpressionLineage(
      rel: PelagoUnpack,
      mq: RelMetadataQuery,
      outputExpression: RexNode
  ): util.Set[RexNode] = mq.getExpressionLineage(rel.getInput, outputExpression)

  def getExpressionLineage(
      rel: PelagoDeviceCross,
      mq: RelMetadataQuery,
      outputExpression: RexNode
  ): util.Set[RexNode] = mq.getExpressionLineage(rel.getInput, outputExpression)

  private[metadata] def findFieldIdx(rel: PelagoTableScan, ind: Int) = { //    int i = 0;
//    for (int f: rel.fields()){
//      if (f == ind) return i;
//      ++i;
//    }
    rel.fields(ind)
  }

  def getExpressionLineage(
      rel: PelagoTableScan,
      mq: RelMetadataQuery,
      outputExpression: RexNode
  ): util.Set[RexNode] = {
    val rexBuilder = rel.getCluster.getRexBuilder
    val inputExtraFields = new util.LinkedHashSet[RelDataTypeField]
    val inputFinder = new RelOptUtil.InputFinder(inputExtraFields)
    outputExpression.accept(inputFinder)
    val inputFieldsUsed = inputFinder.build
// Infer column origin expressions for given references
    val mapping = new util.LinkedHashMap[RexInputRef, util.Set[RexNode]]
    import scala.collection.JavaConversions._
    for (idx <- inputFieldsUsed) {
      val tableIdx = findFieldIdx(rel, idx)
      val inputRef = RexTableInputRef.of(
        RexTableInputRef.RelTableRef.of(rel.getTable, 0),
        RexInputRef.of(tableIdx, rel.getTable.getRowType.getFieldList)
      )
      val ref = RexInputRef.of(idx, rel.getRowType.getFieldList)
      mapping.put(ref, ImmutableSet.of(inputRef))
    }
// Return result
    PelagoRelMdExpressionLineage.createAllPossibleExpressions(
      rexBuilder,
      outputExpression,
      mapping
    )
  }

  def getExpressionLineage(
      rel: PelagoRouter,
      mq: RelMetadataQuery,
      outputExpression: RexNode
  ): util.Set[RexNode] = mq.getExpressionLineage(rel.getInput, outputExpression)
}
