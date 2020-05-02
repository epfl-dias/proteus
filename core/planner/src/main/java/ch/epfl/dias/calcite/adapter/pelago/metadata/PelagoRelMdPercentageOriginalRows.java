package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdPercentageOriginalRows;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;

public class PelagoRelMdPercentageOriginalRows
    implements MetadataHandler<BuiltInMetadata.PercentageOriginalRows> {
  private static final PelagoRelMdPercentageOriginalRows INSTANCE =
      new PelagoRelMdPercentageOriginalRows();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.PERCENTAGE_ORIGINAL_ROWS.method, PelagoRelMdPercentageOriginalRows.INSTANCE),
              RelMdPercentageOriginalRows.SOURCE));

  //~ Methods ----------------------------------------------------------------

  private PelagoRelMdPercentageOriginalRows() {}

  public MetadataDef<BuiltInMetadata.PercentageOriginalRows> getDef() {
    return BuiltInMetadata.PercentageOriginalRows.DEF;
  }

  public Double getPercentageOriginalRows(PelagoUnpack rel, RelMetadataQuery mq) {
    return mq.getPercentageOriginalRows(rel.getInput());
  }

  public Double getPercentageOriginalRows(PelagoPack rel, RelMetadataQuery mq) {
    return mq.getPercentageOriginalRows(rel.getInput());
  }

  public Double getPercentageOriginalRows(PelagoRouter rel, RelMetadataQuery mq) {
    return mq.getPercentageOriginalRows(rel.getInput());
  }

  public Double getPercentageOriginalRows(PelagoDeviceCross rel, RelMetadataQuery mq) {
    return mq.getPercentageOriginalRows(rel.getInput());
  }
}

// End RelMdPercentageOriginalRows.java
