package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.plan.volcano.RelSubset;
import org.apache.calcite.rel.BiRel;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
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
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistributionTraitDef;

import java.util.List;

public class PelagoRelMdHetDistribution implements MetadataHandler<HetDistribution> {
  private static final PelagoRelMdHetDistribution INSTANCE = new PelagoRelMdHetDistribution();

  public static final RelMetadataProvider SOURCE = ReflectiveRelMetadataProvider.reflectiveSource(
          HetDistribution.method, PelagoRelMdHetDistribution.INSTANCE
  );

  public MetadataDef<HetDistribution> getDef() {
    return HetDistribution.DEF;
  }

  public RelHetDistribution hetDistribution(RelNode rel, RelMetadataQuery mq) {
    return RelHetDistributionTraitDef.INSTANCE.getDefault();
  }

  public RelHetDistribution hetDistribution(RelSubset rel, RelMetadataQuery mq) {
    return rel.getTraitSet().getTrait(RelHetDistributionTraitDef.INSTANCE);
  }

  public RelHetDistribution hetDistribution(SingleRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).hetDistribution(rel.getInput());
  }

  public RelHetDistribution hetDistribution(PelagoUnpack rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).hetDistribution(rel.getInput());
  }

  public RelHetDistribution hetDistribution(BiRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).hetDistribution(rel.getRight());
  }

  public RelHetDistribution hetDistribution(PelagoTableScan scan, RelMetadataQuery mq){
    return RelHetDistribution.SINGLETON;
  }

  public RelHetDistribution hetDistribution(PelagoSplit split, RelMetadataQuery mq) {
    return split.hetdistribution();
  }

  public RelHetDistribution hetDistribution(PelagoUnion union, RelMetadataQuery mq) {
    return RelHetDistribution.SINGLETON;
  }

}
