package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.DeviceCross;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdDistribution;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;

public class PelagoRelMdDistribution implements MetadataHandler<BuiltInMetadata.Distribution> {
  private static final PelagoRelMdDistribution INSTANCE = new PelagoRelMdDistribution();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  BuiltInMethod.DISTRIBUTION.method, PelagoRelMdDistribution.INSTANCE),
              RelMdDistribution.SOURCE));

  public MetadataDef<BuiltInMetadata.Distribution> getDef() {
    return BuiltInMetadata.Distribution.DEF;
  }

  public RelDistribution distribution(RelNode rel, RelMetadataQuery mq) {
    RelDistribution dtype = rel.getTraitSet().getTrait(RelDistributionTraitDef.INSTANCE); //TODO: is this safe ? or can it cause an inf loop?
    if (dtype != null) return dtype;
    return RelDistributions.SINGLETON;
  }

  public RelDistribution distribution(PelagoTableScan scan, RelMetadataQuery mq){
    return scan.getDistribution();
  }

  public RelDistribution distribution(DeviceCross devcross, RelMetadataQuery mq) {
    return mq.distribution(devcross.getInput());
  }

}
