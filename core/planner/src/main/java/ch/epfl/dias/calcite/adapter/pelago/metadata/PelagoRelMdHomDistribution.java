package ch.epfl.dias.calcite.adapter.pelago.metadata;

import ch.epfl.dias.calcite.adapter.pelago.*;
import org.apache.calcite.plan.volcano.RelSubset;
import org.apache.calcite.rel.BiRel;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.util.mapping.Mappings;

import java.util.List;

public class PelagoRelMdHomDistribution implements MetadataHandler<HomDistribution> {
  private static final PelagoRelMdHomDistribution INSTANCE = new PelagoRelMdHomDistribution();

  public static final RelMetadataProvider SOURCE = ReflectiveRelMetadataProvider.reflectiveSource(
          HomDistribution.method, PelagoRelMdHomDistribution.INSTANCE
  );

  public MetadataDef<HomDistribution> getDef() {
    return HomDistribution.DEF;
  }

  public RelHomDistribution homDistribution(RelSubset rel, RelMetadataQuery mq) {
    return rel.getTraitSet().getTrait(RelHomDistributionTraitDef.INSTANCE);
  }

  public RelHomDistribution homDistribution(SingleRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).homDistribution(rel.getInput());
  }

  public RelHomDistribution homDistribution(PelagoUnpack rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).homDistribution(rel.getInput());
  }

  public RelHomDistribution homDistribution(BiRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).homDistribution(rel.getRight());
  }

  public RelHomDistribution homDistribution(PelagoSplit split, RelMetadataQuery mq) {
    return split.homdistribution();
  }

  public RelHomDistribution homDistribution(PelagoUnion union, RelMetadataQuery mq) {
    return RelHomDistribution.SINGLE;
  }

  public RelHomDistribution homDistribution(PelagoTableModify mod, RelMetadataQuery mq) {
    var ret = ((PelagoRelMetadataQuery) mq).homDistribution(mod.getInput());
    assert(ret == RelHomDistribution.SINGLE);
    return ret;
  }

  public RelHomDistribution homDistribution(RelNode rel, RelMetadataQuery mq) {
    RelHomDistribution dtype = rel.getTraitSet().getTrait(RelHomDistributionTraitDef.INSTANCE); //TODO: is this safe ? or can it cause an inf loop?
    if (dtype != null) return dtype;
    return RelHomDistribution.SINGLE;
  }

  public RelHomDistribution homDistribution(PelagoRouter router, RelMetadataQuery mq){
//    System.out.println(scan.getDistribution());
    return router.getHomDistribution();
  }

  public RelHomDistribution homDistribution(PelagoTableScan scan, RelMetadataQuery mq){
//    System.out.println(scan.getDistribution());
    return scan.getHomDistribution();
  }

  public RelHomDistribution homDistribution(PelagoDeviceCross devcross, RelMetadataQuery mq) {
//    System.out.println("asdasd");
    return((PelagoRelMetadataQuery) mq).homDistribution(devcross.getInput());
  }

  public static RelHomDistribution project(RelMetadataQuery mq, RelNode input, List<? extends RexNode> projects) {
//    return mq.distribution(input);
//    Mappings.TargetMapping mapping = Project.getPartialMapping(input.getRowType().getFieldCount(), projects);
//    return inputDistribution.apply(mapping);
    return ((PelagoRelMetadataQuery) mq).homDistribution(input);
  }

}
