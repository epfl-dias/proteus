package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.JaninoRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;

import com.google.common.base.Preconditions;

import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

public class PelagoRelMetadataQuery extends RelMetadataQuery {
  private DeviceType.Handler deviceTypeHandler;

  protected PelagoRelMetadataQuery(JaninoRelMetadataProvider metadataProvider,
      RelMetadataQuery prototype) {
    super(metadataProvider, prototype);
    deviceTypeHandler = initialHandler(DeviceType.Handler.class);
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
   * @return List of sorted column combinations, or
   * null if not enough information is available to make that determination
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

}
