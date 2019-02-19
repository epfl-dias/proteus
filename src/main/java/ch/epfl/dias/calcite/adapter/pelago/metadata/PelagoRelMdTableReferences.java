package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdTableReferences;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexTableInputRef;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;

import java.util.Set;

public class PelagoRelMdTableReferences
    implements MetadataHandler<BuiltInMetadata.TableReferences> {
  private static final PelagoRelMdTableReferences INSTANCE = new PelagoRelMdTableReferences();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.TABLE_REFERENCES.method, PelagoRelMdTableReferences.INSTANCE),
              RelMdTableReferences.SOURCE));

  protected PelagoRelMdTableReferences() {}

  public MetadataDef<BuiltInMetadata.TableReferences> getDef() {
    return BuiltInMetadata.TableReferences.DEF;
  }

  public Set<RexTableInputRef.RelTableRef> getTableReferences(PelagoUnpack rel, RelMetadataQuery mq) {
    return mq.getTableReferences(rel.getInput());
  }

  public Set<RexTableInputRef.RelTableRef> getTableReferences(PelagoPack rel, RelMetadataQuery mq) {
    return mq.getTableReferences(rel.getInput());
  }

  public Set<RexTableInputRef.RelTableRef> getTableReferences(PelagoDeviceCross rel, RelMetadataQuery mq) {
    return mq.getTableReferences(rel.getInput());
  }
}
