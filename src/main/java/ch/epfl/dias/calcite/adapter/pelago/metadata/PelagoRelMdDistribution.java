package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.BiRel;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.DeviceCross;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMdDistribution;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.util.BuiltInMethod;
import org.apache.calcite.util.mapping.Mappings;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;

import java.util.List;

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

  public RelDistribution distribution(BiRel rel, RelMetadataQuery mq) {
    return mq.distribution(rel.getRight());
  }

  public RelDistribution distribution(PelagoTableScan scan, RelMetadataQuery mq){
//    System.out.println(scan.getDistribution());
    return scan.getDistribution();
  }

  public RelDistribution distribution(DeviceCross devcross, RelMetadataQuery mq) {
//    System.out.println("asdasd");
    return mq.distribution(devcross.getInput());
  }

  public static RelDistribution project(RelMetadataQuery mq, RelNode input, List<? extends RexNode> projects) {
    return mq.distribution(input);
//    Mappings.TargetMapping mapping = Project.getPartialMapping(input.getRowType().getFieldCount(), projects);
//    return inputDistribution.apply(mapping);
  }

}
