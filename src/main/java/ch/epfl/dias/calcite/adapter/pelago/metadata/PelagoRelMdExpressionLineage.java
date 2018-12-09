package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdExpressionLineage;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;

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
}
