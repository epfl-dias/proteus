package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.EquiJoin;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.CyclicMetadataException;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
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

public class PelagoRelMdRowCount implements MetadataHandler<BuiltInMetadata.RowCount> {
  private static final PelagoRelMdRowCount INSTANCE = new PelagoRelMdRowCount();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.ROW_COUNT.method, PelagoRelMdRowCount.INSTANCE),
              RelMdRowCount.SOURCE));

  private final RelMdRowCount def = new RelMdRowCount();

  protected PelagoRelMdRowCount(){}

  @Override public MetadataDef<BuiltInMetadata.RowCount> getDef() {
    return BuiltInMetadata.RowCount.DEF;
  }

  public Double getRowCount(PelagoUnnest rel, RelMetadataQuery mq) {
    return rel.estimateRowCount(mq);
  }

  public Double getRowCount(PelagoPack rel, RelMetadataQuery mq) {
    return Math.ceil(mq.getRowCount(rel.getInput()) / (1024*1024.0));
  }

  public Double getRowCount(Aggregate rel, RelMetadataQuery mq) {
    if (rel.getGroupCount() == 0) return 1.0;
    return def.getRowCount(rel, mq); // groupby's are generally very selective
  }

  public Double getRowCount(PelagoUnpack rel, RelMetadataQuery mq) {
    return mq.getRowCount(rel.getInput()) * (1024*1024.0);
  }

  public Double getRowCount(PelagoRouter rel, RelMetadataQuery mq) {
    return rel.estimateRowCount(mq);//mq.getRowCount(rel.getInput()) / 2;
  }

  public Double getRowCount(PelagoUnion rel, RelMetadataQuery mq) {
    return mq.getRowCount(rel.getInput(0)) + mq.getRowCount(rel.getInput(1));
  }

  public Double getRowCount(PelagoDeviceCross rel, RelMetadataQuery mq) {
    return mq.getRowCount(rel.getInput());
  }

  public Double getRowCount(PelagoTableScan rel, RelMetadataQuery mq) {
    double rc = rel.getTable().getRowCount();
    if (rel.getTraitSet().containsIfApplicable(RelPacking.Packed)) rc /= 1024;
    return rc;
  }



  public Double getRowCount(Join rel, RelMetadataQuery mq) {
    Double leftRows = mq.getRowCount(rel.getLeft());
    Double rightRows = mq.getRowCount(rel.getRight());
    Double rows = Math.max(leftRows, rightRows);
    RelNode small = (leftRows < rightRows) ? rel.getLeft() : rel.getRight();
    Double sel = mq.getPercentageOriginalRows(small);
    if (sel == null) return rows;
    return rows * sel;
  }
}
