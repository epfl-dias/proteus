package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.EquiJoin;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdMaxRowCount;
import org.apache.calcite.rel.metadata.RelMdRowCount;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnnest;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;
import ch.epfl.dias.calcite.adapter.pelago.costs.CostModel;

public class PelagoRelMdMaxRowCount implements MetadataHandler<BuiltInMetadata.MaxRowCount> {
  private static final PelagoRelMdMaxRowCount INSTANCE = new PelagoRelMdMaxRowCount();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.MAX_ROW_COUNT.method, PelagoRelMdMaxRowCount.INSTANCE),
              RelMdMaxRowCount.SOURCE));

  private final RelMdMaxRowCount def = new RelMdMaxRowCount();

  protected PelagoRelMdMaxRowCount(){}

  @Override public MetadataDef<BuiltInMetadata.MaxRowCount> getDef() {
    return BuiltInMetadata.MaxRowCount.DEF;
  }

  public Double getMaxRowCount(PelagoPack rel, RelMetadataQuery mq) {
    return Math.ceil(mq.getMaxRowCount(rel.getInput()) / CostModel.blockSize());
  }

  public Double getMaxRowCount(PelagoUnpack rel, RelMetadataQuery mq) {
    return mq.getMaxRowCount(rel.getInput()) * CostModel.blockSize();
  }

  public Double getMaxRowCount(PelagoUnnest rel, RelMetadataQuery mq) {
    return Double.POSITIVE_INFINITY;
  }

  public Double getMaxRowCount(PelagoRouter rel, RelMetadataQuery mq) {
    return mq.getMaxRowCount(rel.getInput());//mq.getRowCount(rel.getInput()) / 2;
  }

  public Double getMaxRowCount(PelagoUnion rel, RelMetadataQuery mq) {
    return mq.getMaxRowCount(rel.getInput(0)) + mq.getMaxRowCount(rel.getInput(1));
  }

  public Double getMaxRowCount(PelagoDeviceCross rel, RelMetadataQuery mq) {
    return mq.getMaxRowCount(rel.getInput());
  }

  public Double getMaxRowCount(PelagoTableScan rel, RelMetadataQuery mq) {
    double rc = rel.getTable().getRowCount();
    if (rel.getTraitSet().containsIfApplicable(RelPacking.Packed)) rc /= CostModel.blockSize();
    return rc;
  }
}
