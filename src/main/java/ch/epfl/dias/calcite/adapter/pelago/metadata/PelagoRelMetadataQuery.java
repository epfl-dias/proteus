package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.JaninoRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;

import com.google.common.base.Preconditions;

import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

public class PelagoRelMetadataQuery extends RelMetadataQuery {
  private DeviceType.Handler deviceTypeHandler;
  private Packing.Handler    packingHandler;

  protected PelagoRelMetadataQuery(JaninoRelMetadataProvider metadataProvider,
                                   RelMetadataQuery prototype) {
    super(metadataProvider, prototype);
    deviceTypeHandler = initialHandler(DeviceType.Handler.class);
    packingHandler    = initialHandler(Packing   .Handler.class);
  }

  public static PelagoRelMetadataQuery instance() {
    return new PelagoRelMetadataQuery(THREAD_PROVIDERS.get(), EMPTY);
  }
  /**
   * Returns the
   * {@link DeviceType#deviceType()}
   * statistic.
   *
   * @param rel         the relational expression
   */
  public RelDeviceType deviceType(RelNode rel) {
    for (;;) {
      try {
        return deviceTypeHandler.deviceType(rel, this);
      } catch (JaninoRelMetadataProvider.NoHandler e) {
        deviceTypeHandler = revise(e.relClass, DeviceType.DEF);
      }
    }
  }
  /**
   * Returns the
   * {@link Packing#packing()}
   * statistic.
   *
   * @param rel         the relational expression
   */
  public RelPacking packing(RelNode rel) {
    for (;;) {
      try {
        return packingHandler.packing(rel, this);
      } catch (JaninoRelMetadataProvider.NoHandler e) {
        packingHandler = revise(e.relClass, Packing.DEF);
      }
    }
  }
}
