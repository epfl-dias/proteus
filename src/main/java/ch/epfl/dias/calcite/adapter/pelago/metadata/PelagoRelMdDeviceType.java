package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.BiRel;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Exchange;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.SetOp;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.rules.MultiJoin;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoToEnumerableConverter;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;

import java.util.List;

public class PelagoRelMdDeviceType implements MetadataHandler<DeviceType> {
  private static final PelagoRelMdDeviceType INSTANCE = new PelagoRelMdDeviceType();

  public static final RelMetadataProvider SOURCE =
      ChainedRelMetadataProvider.of(
          ImmutableList.of(
              ReflectiveRelMetadataProvider.reflectiveSource(
                  DeviceType.method, PelagoRelMdDeviceType.INSTANCE),
              RelMdDeviceType.SOURCE));

  public MetadataDef<DeviceType> getDef() {
    return DeviceType.DEF;
  }

  public RelDeviceType deviceType(PelagoTableScan scan, RelMetadataQuery mq) {
//    System.out.println(scan.getDeviceType());
    return scan.getDeviceType();
  }
  public RelDeviceType deviceType(SingleRel rel, PelagoRelMetadataQuery mq) {
    return mq.deviceType(rel.getInput());
  }

  public RelDeviceType deviceType(Project rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getInput());
  }

  public RelDeviceType deviceType(BiRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getRight());
  }

  public RelDeviceType deviceType(MultiJoin rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getInputs().get(0));
  }

  public RelDeviceType deviceType(SetOp rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getInputs().get(0));
  }

  public RelDeviceType deviceType(Values values, RelMetadataQuery mq) {
    return RelDeviceType.ANY;
  }

  public RelDeviceType deviceType(HepRelVertex rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getCurrentRel());
  }

  public RelDeviceType deviceType(RelNode rel, RelMetadataQuery mq) {
    RelDeviceType dtype = rel.getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE); //TODO: is this safe ? or can it cause an inf loop?
    if (dtype != null) return dtype;
    return RelDeviceType.X86_64;
  }

  public RelDeviceType deviceType(PelagoDeviceCross devcross, RelMetadataQuery mq) {
    return devcross.getDeviceType();
  }

  public RelDeviceType deviceType(Exchange xchange, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(xchange.getInput());
  }

  public RelDeviceType pelagoToEnumberable(RelMetadataQuery mq, RelNode input){
    return ((PelagoRelMetadataQuery) mq).deviceType(input);
  }

  public RelDeviceType deviceType(PelagoToEnumerableConverter conv, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(conv);
  }

  /** Helper method to determine a
   * {@link Aggregate}'s deviceType. */
  public static RelDeviceType aggregate(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).deviceType(input);
  }


  /** Helper method to determine a {@link Project}'s collation. */
  public static RelDeviceType project(RelMetadataQuery mq, RelNode input,
      List<? extends RexNode> projects) {
    final RelDeviceType inputdeviceType = ((PelagoRelMetadataQuery) mq).deviceType(input);
//    final Mappings.TargetMapping mapping =
//        Project.getPartialMapping(input.getRowType().getFieldCount(),
//            projects);
    return inputdeviceType; //.apply(mapping); // TODO: Should we do something here ?
  }
}
