package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.util.BuiltInMethod;

public class PelagoRelMdDistribution implements MetadataHandler<BuiltInMetadata.Distribution> {
  private static final PelagoRelMdDistribution INSTANCE = new PelagoRelMdDistribution();

  public static final RelMetadataProvider SOURCE =
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.DISTRIBUTION.method, PelagoRelMdDistribution.INSTANCE);

  public MetadataDef<BuiltInMetadata.Distribution> getDef() {
    return BuiltInMetadata.Distribution.DEF;
  }

  public RelDistribution distribution(RelNode rel, RelMetadataQuery mq) {
    assert(false);
    return null;
  }
}
