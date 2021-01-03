package ch.epfl.dias.calcite.adapter.pelago.metadata;

import com.google.common.collect.ImmutableRangeSet;
import com.google.common.collect.Range;
import org.apache.calcite.plan.RelOptUtil;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdSelectivity;
import org.apache.calcite.rel.metadata.RelMdUtil;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.util.*;

import com.google.common.collect.ImmutableList;
import ch.epfl.dias.calcite.adapter.pelago.schema.PelagoTable;

import java.util.*;

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


  private static Double guessEqSelectivity(RexTableInputRef attr) {
    PelagoTable table = attr.getTableRef().getTable().unwrap(PelagoTable.class);

    Double distinctValues = table.getDistrinctValues(ImmutableBitSet.of(attr.getIndex()));
    if (distinctValues == null) return null;
    return 1 / distinctValues;
  }

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

    return guessEqSelectivity((RexTableInputRef) a);
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

  private static class Estimator <C extends Comparable<C>> implements RangeSets.Consumer<C> {
    private double ret = 0;
    private final ImmutableList.Builder<RexNode> remaining = ImmutableList.builder();
    private final RexBuilder rexBuilder;
    private final RelDataType type;
    private final PelagoTable table;
    private final RexTableInputRef attr;
    private final RexNode ref;

    Estimator(RexBuilder rexBuilder, RelDataType type, RexTableInputRef attr, RexNode ref){
      this.rexBuilder = rexBuilder;
      this.type = type;
      this.table = attr.getTableRef().getTable().unwrap(PelagoTable.class);
      this.attr = attr;
      this.ref = ref;
    }

    public Double getResult() {
      var rems = remaining.build();
      if (!rems.isEmpty()) {
        ret += RelMdUtil.guessSelectivity(RexUtil.composeConjunction(rexBuilder, rems));
      }
      return ret;
    }

    private RexLiteral getLiteral(C v) {
      return (RexLiteral) rexBuilder.makeLiteral(v, type, false, false);
    }

    private Double getPercentile(C v) {
      return table.getPercentile(ImmutableBitSet.of(attr.getIndex()), getLiteral(v), rexBuilder);
    }

    @Override
    public void all() {
      ret += 1;
    }

    @Override
    public void atLeast(C lower) {
      greaterThan(lower);
    }

    @Override
    public void atMost(C upper) {
      lessThan(upper);
    }

    @Override
    public void greaterThan(C lower) {
      Double local_sel = getPercentile(lower);
      if (local_sel == null) {
        remaining.add(rexBuilder.makeCall(SqlStdOperatorTable.GREATER_THAN, ref, getLiteral(lower)));
      } else {
        ret += 1 - local_sel;
      }
    }

    @Override
    public void lessThan(C upper) {
      Double local_sel = getPercentile(upper);
      if (local_sel == null) {
        remaining.add(rexBuilder.makeCall(SqlStdOperatorTable.LESS_THAN, ref, getLiteral(upper)));
      } else {
        ret += local_sel;
      }
    }

    @Override
    public void singleton(C value) {
      var eq = guessEqSelectivity(attr);
      if (eq == null) {
        remaining.add(rexBuilder.makeCall(SqlStdOperatorTable.EQUALS, ref, getLiteral(value)));
      } else {
        ret += eq;
      }
    }

    @Override
    public void closed(C lower, C upper) {
      var up = getPercentile(upper);
      var dn = getPercentile(lower);
      if (up == null || dn == null) {
        remaining.add(rexBuilder.makeCall(SqlStdOperatorTable.SEARCH, ref, rexBuilder.makeSearchArgumentLiteral(
            Sarg.of(false, ImmutableRangeSet.of(Range.closed(lower, upper))),
            ref.getType()
        )));
      } else {
        ret += getPercentile(upper) - getPercentile(lower);
      }
    }

    @Override
    public void closedOpen(C lower, C upper) {
      closed(lower, upper);
    }

    @Override
    public void openClosed(C lower, C upper) {
      closed(lower, upper);
    }

    @Override
    public void open(C lower, C upper) {
      closed(lower, upper);
    }
  }

  public <C extends Comparable<C>> Double guessBetweenSelectivity(RelNode rel, RelMetadataQuery mq, RexNode predicate) {
    if (!(predicate instanceof RexCall)) return null;
    if (!predicate.isA(SqlKind.SEARCH)) return null;

    RexCall call = (RexCall) predicate;

    RexTableInputRef attr = getReferencedAttr((RexCall) predicate, mq, rel);
    if (attr == null) return null;


    final RexBuilder rexBuilder = rel.getCluster().getRexBuilder();
    final RexNode ref = call.operands.get(0);
    final RexLiteral literal = (RexLiteral) call.operands.get(1);
    final Sarg<C> sarg = literal.getValueAs(Sarg.class);
    double ret = 0.0;

    if (sarg.containsNull) {
      ret += 0.1; // TODO: ask statistics for percentage of null values
    }
    if (sarg.isComplementedPoints()) {
      // Generate 'ref <> value1 AND ... AND ref <> valueN'

      // TODO: expand with more detailed statistics, as for now it assumes equal probability
      //  for all values and all values to be part of the domain
      ret += 1 - sarg.rangeSet.asRanges().size() * guessEqSelectivity(attr);
    } else {
      var consumer = new Estimator<C>(rexBuilder, literal.getType(), attr, ref);
      RangeSets.forEach(sarg.rangeSet, consumer);
      ret += consumer.getResult();
    }

    return Math.min(Math.max(ret, 0), 1);
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
      } else if (pred.isA(SqlKind.SEARCH)) {
        local_sel = guessBetweenSelectivity(rel, mq, pred);
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
