package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.prepare.RelOptTableImpl;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.EquiJoin;
import org.apache.calcite.rel.core.Union;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdRowCount;
import org.apache.calcite.rel.metadata.RelMdSelectivity;
import org.apache.calcite.rel.metadata.RelMdUtil;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.util.BuiltInMethod;
import org.apache.calcite.util.ImmutableBitSet;
import org.apache.calcite.util.Pair;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSchema;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTable;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnnest;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;
import scala.Immutable;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class PelagoRelMdSelectivity implements MetadataHandler<BuiltInMetadata.Selectivity> {
  private static final PelagoRelMdSelectivity INSTANCE = new PelagoRelMdSelectivity();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.SELECTIVITY.method, PelagoRelMdSelectivity.INSTANCE),
              RelMdSelectivity.SOURCE));

  protected PelagoRelMdSelectivity(){}

  @Override public MetadataDef<BuiltInMetadata.Selectivity> getDef() {
    return BuiltInMetadata.Selectivity.DEF;
  }
//
//  public Double getSelectivity(PelagoPack rel, RelMetadataQuery mq,
//      RexNode predicate) {
//    return mq.getSelectivity(rel.getInput(), predicate);
//  }
//
//  public Double getSelectivity(PelagoUnpack rel, RelMetadataQuery mq,
//      RexNode predicate) {
//    return mq.getSelectivity(rel.getInput(), predicate);
//  }
//
//  public Double getSelectivity(PelagoRouter rel, RelMetadataQuery mq,
//      RexNode predicate) {
//    return rel.estimateRowCount(mq);//mq.getRowCount(rel.getInput()) / 2;
//  }
//
//  public Double getSelectivity(PelagoUnion rel, RelMetadataQuery mq,
//      RexNode predicate) {
//    return mq.getRowCount(rel.getInput(0)) + mq.getRowCount(rel.getInput(1));
//  }
//
//  public Double getSelectivity(PelagoDeviceCross rel, RelMetadataQuery mq,
//      RexNode predicate) {
//    return mq.getRowCount(rel.getInput());
//  }


  public Double guessEqSelectivity(RelNode rel, RelMetadataQuery mq, RexNode predicate){
    if (!(predicate instanceof RexCall)) return null;
    if (!predicate.isA(SqlKind.EQUALS)) return null;

    List<RexNode> bi = ((RexCall) predicate).getOperands();
    assert(bi.size() == 2);

    // Check that exactly one of the inputs is a literal
    if ((bi.get(0) instanceof RexLiteral) == (bi.get(1) instanceof RexLiteral)){
      return null;
    }

    RexNode input = (bi.get(1) instanceof RexLiteral) ? bi.get(0) : bi.get(1);

    Set<RexNode> ls = mq.getExpressionLineage(rel, input);

    if (ls == null || ls.isEmpty()) return null;


    // Check that it references a column, as we do not support transformations
    RexNode a = ls.iterator().next();
    if (!(a instanceof RexTableInputRef)) return null;

    RexTableInputRef attr = (RexTableInputRef) a;
    PelagoTable table = attr.getTableRef().getTable().unwrap(PelagoTable.class);

    Double distinctValues = table.getDistrinctValues(ImmutableBitSet.of(attr.getIndex()));
    if (distinctValues == null) return null;
    return 1/distinctValues;
  }

  public static RexTableInputRef getReferencedAttr(RexCall predicate, RelMetadataQuery mq, RelNode rel){
    List<RexNode> bi = predicate.getOperands();
    assert(bi.size() == 2);

    // Check that exactly one of the inputs is a literal
    if ((bi.get(0) instanceof RexLiteral) == (bi.get(1) instanceof RexLiteral)){
      return null;
    }

    RexNode input = (bi.get(1) instanceof RexLiteral) ? bi.get(0) : bi.get(1);
    RexLiteral val = (RexLiteral) ((bi.get(1) instanceof RexLiteral) ? bi.get(1) : bi.get(0));

    Set<RexNode> ls = mq.getExpressionLineage(rel, input);

    if (ls == null || ls.isEmpty()) return null;


    // Check that it references a column, as we do not support transformations
    RexNode a = ls.iterator().next();
    if (!(a instanceof RexTableInputRef)) return null;

    return (RexTableInputRef) a;
  }

  public static RexLiteral getReferencedLiteral(RexCall predicate, RelMetadataQuery mq, RelNode rel){
    List<RexNode> bi = predicate.getOperands();
    assert(bi.size() == 2);

    // Check that exactly one of the inputs is a literal
    if ((bi.get(0) instanceof RexLiteral) == (bi.get(1) instanceof RexLiteral)){
      return null;
    }

    return (RexLiteral) ((bi.get(1) instanceof RexLiteral) ? bi.get(1) : bi.get(0));
  }

  public Double guessCmpSelectivity(RelNode rel, RelMetadataQuery mq, RexNode predicate){
    if (!(predicate instanceof RexCall)) return null;
    if (!predicate.isA(SqlKind.COMPARISON)) return null;

    RexTableInputRef attr = getReferencedAttr((RexCall) predicate, mq, rel);
    RexLiteral lit = getReferencedLiteral((RexCall) predicate, mq, rel);
    if (attr == null) return null;
    PelagoTable table = attr.getTableRef().getTable().unwrap(PelagoTable.class);

    return table.getPercentile(ImmutableBitSet.of(attr.getIndex()), lit, rel.getCluster().getRexBuilder());
  }

  public Double guessSelectivity(RelNode rel, RelMetadataQuery mq, RexNode predicate) {
    if (predicate == null) return null;
    double sel = 1.0;

    ImmutableList.Builder remaining = ImmutableList.builder();
    Map<RexTableInputRef, Pair<Boolean, Double>> bounds = new HashMap<>();

    for (RexNode pred : RelOptUtil.conjunctions(predicate)) {
      Double local_sel = null;

      if (pred.isA(SqlKind.EQUALS)) {
        local_sel = guessEqSelectivity(rel, mq, pred);
      } else if (pred.isA(SqlKind.COMPARISON)) {
        local_sel = guessCmpSelectivity(rel, mq, pred);
        boolean lower_limit = pred.isA(SqlKind.GREATER_THAN) || pred.isA(SqlKind.GREATER_THAN_OR_EQUAL);
//        if (lower_limit) {
//          local_sel = 1 - local_sel;
//        }
        if (local_sel != null){
          RexTableInputRef ref = getReferencedAttr((RexCall) pred, mq, rel);
          if (bounds.containsKey(ref)) {
            Pair<Boolean, Double> other = bounds.remove(ref);
            assert(other.left.booleanValue() != lower_limit);
            local_sel = Math.abs(other.right - local_sel);
          } else {
            bounds.put(ref, new Pair<>(lower_limit, local_sel));
            local_sel = 1.0;
          }
        }
      } else if (pred.isA(SqlKind.OR)) {
        local_sel = ((RexCall) pred).getOperands().stream().map((e) -> guessSelectivity(rel, mq, e)).reduce(0.0, (a, b) -> a+b);
        local_sel = Math.min(Math.max(local_sel, 0), 1);
      }

      if (local_sel == null) {
        remaining.add(pred);
      } else {
        sel *= local_sel;
      }
    }

    for (Pair<Boolean, Double> v: bounds.values()){
      double local_sel = v.right;
      if (v.left) local_sel = 1 - local_sel;
      sel *= local_sel;
    }

    ImmutableList rems = remaining.build();
    if (!rems.isEmpty()) {
      sel *= RelMdUtil.guessSelectivity(RexUtil.composeConjunction(rel.getCluster().getRexBuilder(), remaining.build()));
    }

    return sel;
  }


  public Double getSelectivity(RelNode rel, RelMetadataQuery mq,
      RexNode predicate) {
    Double sel = guessSelectivity(rel, mq, predicate);
    if (sel != null) return sel;

    return RelMdUtil.guessSelectivity(predicate);
  }
}
