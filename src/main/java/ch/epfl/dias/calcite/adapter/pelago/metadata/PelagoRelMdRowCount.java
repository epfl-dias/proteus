package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdRowCount;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.util.BuiltInMethod;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnnest;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

public class PelagoRelMdRowCount implements MetadataHandler<BuiltInMetadata.RowCount> {
    public static final RelMetadataProvider SOURCE =
        ReflectiveRelMetadataProvider.reflectiveSource(
            BuiltInMethod.ROW_COUNT.method, new PelagoRelMdRowCount());

  private final RelMdRowCount def = new RelMdRowCount();

  @Override public MetadataDef<BuiltInMetadata.RowCount> getDef() {
    return BuiltInMetadata.RowCount.DEF;
  }

  public Double getRowCount(PelagoUnnest rel, RelMetadataQuery mq) {
    return rel.estimateRowCount(mq);
  }

  public Double getRowCount(PelagoRouter rel, RelMetadataQuery mq) {
    return rel.estimateRowCount(mq);
  }

  public Double getRowCount(PelagoPack rel, RelMetadataQuery mq) {
    return rel.estimateRowCount(mq);
  }

  public Double getRowCount(Aggregate rel, RelMetadataQuery mq) {
    if (rel.getGroupCount() == 0) return 1.0;
    return def.getRowCount(rel, mq);
  }

  public Double getRowCount(PelagoUnpack rel, RelMetadataQuery mq) {
    return rel.estimateRowCount(mq);
  }

  public Double getRowCount(PelagoDeviceCross rel, RelMetadataQuery mq) {
    return rel.estimateRowCount(mq);
  }

  public Double getRowCount(PelagoTableScan rel, RelMetadataQuery mq) {
    double rc = rel.getTable().getRowCount();
    if (rel.getTraitSet().containsIfApplicable(RelPacking.Packed)) rc /= 1024;
    return rc;
  }
}
