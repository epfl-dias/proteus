package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRouter;
import ch.epfl.dias.calcite.adapter.pelago.PelagoSplit;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnion;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

public class PelagoRelMdComputeDevice implements MetadataHandler<ComputeDevice> {
  private static final PelagoRelMdComputeDevice INSTANCE = new PelagoRelMdComputeDevice();

  public static final RelMetadataProvider SOURCE = ReflectiveRelMetadataProvider.reflectiveSource(
      ComputeDevice.method, PelagoRelMdComputeDevice.INSTANCE
  );

  public MetadataDef<ComputeDevice> getDef() {
    return ComputeDevice.DEF;
  }

  public RelComputeDevice computeType(RelNode rel, RelMetadataQuery mq) {
    if (rel.getTraitSet().containsIfApplicable(RelDeviceType.X86_64)){
      return RelComputeDevice.X86_64;
    } else if (rel.getTraitSet().containsIfApplicable(RelDeviceType.NVPTX)){
      return RelComputeDevice.NVPTX;
    }
    return RelComputeDevice.from(rel.getInputs().stream().<RelComputeDevice>map((e) -> ((PelagoRelMetadataQuery) mq).computeType(e)));
  }

  public RelComputeDevice computeType(PelagoTableScan scan, RelMetadataQuery mq){
    //if the scan is producing tuples, then it actually runs on the device its DeviceTypeTrait specifies
    if (((PelagoRelMetadataQuery) mq).packing(scan).satisfies(RelPacking.UnPckd)){
      if (scan.getDeviceType().satisfies(RelDeviceType.X86_64)){
        return RelComputeDevice.X86_64;
      } else if (scan.getDeviceType().satisfies(RelDeviceType.NVPTX)) {
        return RelComputeDevice.NVPTX;
      }
    }
    return RelComputeDevice.NONE;
  }

  public RelComputeDevice computeType(PelagoSplit split, RelMetadataQuery mq) {
    return RelComputeDevice.NONE;
  }

  public RelComputeDevice computeType(PelagoUnion split, RelMetadataQuery mq) {
    return RelComputeDevice.NONE;
  }

  public RelComputeDevice computeType(PelagoRouter rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).computeType(rel.getInput());
  }

  public RelComputeDevice computeType(PelagoDeviceCross rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).computeType(rel.getInput());
  }
}
