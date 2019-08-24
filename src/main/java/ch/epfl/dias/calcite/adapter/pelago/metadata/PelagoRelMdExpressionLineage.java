package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdExpressionLineage;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexShuttle;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.util.BuiltInMethod;
import org.apache.calcite.util.ImmutableBitSet;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import org.codehaus.commons.nullanalysis.Nullable;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class PelagoRelMdExpressionLineage implements MetadataHandler<BuiltInMetadata.ExpressionLineage> {
  private static final PelagoRelMdExpressionLineage INSTANCE = new PelagoRelMdExpressionLineage();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.EXPRESSION_LINEAGE.method, PelagoRelMdExpressionLineage.INSTANCE),
              RelMdExpressionLineage.SOURCE));

  @Override public MetadataDef<BuiltInMetadata.ExpressionLineage> getDef() {
    return BuiltInMetadata.ExpressionLineage.DEF;
  }

  public Set<RexNode> getExpressionLineage(PelagoPack rel, RelMetadataQuery mq, RexNode outputExpression){
    return mq.getExpressionLineage(rel.getInput(), outputExpression);
  }

  public Set<RexNode> getExpressionLineage(PelagoUnpack rel, RelMetadataQuery mq, RexNode outputExpression){
    return mq.getExpressionLineage(rel.getInput(), outputExpression);
  }

  public Set<RexNode> getExpressionLineage(PelagoDeviceCross rel, RelMetadataQuery mq, RexNode outputExpression){
    return mq.getExpressionLineage(rel.getInput(), outputExpression);
  }

  private static Set<RexNode> createAllPossibleExpressions(RexBuilder rexBuilder,
      RexNode expr, ImmutableBitSet predFieldsUsed, Map<RexInputRef, Set<RexNode>> mapping,
      Map<RexInputRef, RexNode> singleMapping) {
    final RexInputRef inputRef = mapping.keySet().iterator().next();
    final Set<RexNode> replacements = mapping.remove(inputRef);
    Set<RexNode> result = new HashSet<>();
    assert !replacements.isEmpty();
    if (predFieldsUsed.indexOf(inputRef.getIndex()) != -1) {
      for (RexNode replacement : replacements) {
        singleMapping.put(inputRef, replacement);
        createExpressions(rexBuilder, expr, predFieldsUsed, mapping, singleMapping, result);
        singleMapping.remove(inputRef);
      }
    } else {
      createExpressions(rexBuilder, expr, predFieldsUsed, mapping, singleMapping, result);
    }
    mapping.put(inputRef, replacements);
    return result;
  }

  /**
   * Replaces expressions with their equivalences. Note that we only have to
   * look for RexInputRef.
   */
  private static class RexReplacer extends RexShuttle {
    private final Map<RexInputRef, RexNode> replacementValues;

    RexReplacer(Map<RexInputRef, RexNode> replacementValues) {
      this.replacementValues = replacementValues;
    }

    @Override public RexNode visitInputRef(RexInputRef inputRef) {
      return replacementValues.get(inputRef);
    }
  }

  private static ImmutableBitSet extractInputRefs(RexNode expr) {
    final Set<RelDataTypeField> inputExtraFields = new LinkedHashSet<>();
    final RelOptUtil.InputFinder inputFinder = new RelOptUtil.InputFinder(inputExtraFields);
    expr.accept(inputFinder);
    return inputFinder.inputBitSet.build();
  }

  @Nullable protected static Set<RexNode> createAllPossibleExpressions(RexBuilder rexBuilder,
      RexNode expr, Map<RexInputRef, Set<RexNode>> mapping) {
    // Extract input fields referenced by expression
    final ImmutableBitSet predFieldsUsed = extractInputRefs(expr);

    if (predFieldsUsed.isEmpty()) {
      // The unique expression is the input expression
      return ImmutableSet.of(expr);
    }

    try {
      return createAllPossibleExpressions(rexBuilder, expr, predFieldsUsed, mapping,
          new HashMap<>());
    } catch (UnsupportedOperationException e) {
      // There may be a RexNode unsupported by RexCopier, just return null
      return null;
    }
  }

  private static void createExpressions(RexBuilder rexBuilder,
      RexNode expr, ImmutableBitSet predFieldsUsed, Map<RexInputRef, Set<RexNode>> mapping,
      Map<RexInputRef, RexNode> singleMapping, Set<RexNode> result) {
    if (mapping.isEmpty()) {
      final RexReplacer replacer = new RexReplacer(singleMapping);
      final List<RexNode> updatedPreds = new ArrayList<>(
          RelOptUtil.conjunctions(
              rexBuilder.copy(expr)));
      replacer.mutate(updatedPreds);
      result.addAll(updatedPreds);
    } else {
      result.addAll(
          createAllPossibleExpressions(
              rexBuilder, expr, predFieldsUsed, mapping, singleMapping));
    }
  }

  int findFieldIdx(PelagoTableScan rel, int ind){
//    int i = 0;
//    for (int f: rel.fields()){
//      if (f == ind) return i;
//      ++i;
//    }
    return rel.fields()[ind];
  }

  public Set<RexNode> getExpressionLineage(PelagoTableScan rel, RelMetadataQuery mq, RexNode outputExpression){
    final RexBuilder rexBuilder = rel.getCluster().getRexBuilder();

    final Set<RelDataTypeField> inputExtraFields = new LinkedHashSet<>();
    final RelOptUtil.InputFinder inputFinder = new RelOptUtil.InputFinder(inputExtraFields);
    outputExpression.accept(inputFinder);

    // Extract input fields referenced by expression
    final ImmutableBitSet inputFieldsUsed = inputFinder.inputBitSet.build();

    // Infer column origin expressions for given references
    final Map<RexInputRef, Set<RexNode>> mapping = new LinkedHashMap<>();
    for (int idx : inputFieldsUsed) {
      int tableIdx = findFieldIdx(rel, idx);
      final RexNode inputRef = RexTableInputRef.of(
          RexTableInputRef.RelTableRef.of(rel.getTable(), 0),
          RexInputRef.of(tableIdx, rel.getTable().getRowType().getFieldList()));
      final RexInputRef ref = RexInputRef.of(idx, rel.getRowType().getFieldList());
      mapping.put(ref, ImmutableSet.of(inputRef));
    }

    // Return result
    return createAllPossibleExpressions(rexBuilder, outputExpression, mapping);
  }

  public Set<RexNode> getExpressionLineage(PelagoRouter rel, RelMetadataQuery mq, RexNode outputExpression){
    return mq.getExpressionLineage(rel.getInput(), outputExpression);
  }
}
