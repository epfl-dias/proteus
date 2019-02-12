package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.rel.metadata.ChainedRelMetadataProvider;
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider;

import com.google.common.collect.ImmutableList;

public class PelagoRelMetadataProvider extends ChainedRelMetadataProvider {
  public static final PelagoRelMetadataProvider INSTANCE =
      new PelagoRelMetadataProvider();

  //~ Constructors -----------------------------------------------------------

  /**
   * Creates a new default provider. This provider defines "catch-all"
   * handlers for generic RelNodes, so it should always be given lowest
   * priority when chaining.
   *
   * <p>Use this constructor only from a sub-class. Otherwise use the singleton
   * instance, {@link #INSTANCE}.
   */
  protected PelagoRelMetadataProvider() {
    super(
        ImmutableList.of(
            PelagoRelMdDeviceType       .SOURCE,
            PelagoRelMdComputeDevice    .SOURCE,
            PelagoRelMdDistribution     .SOURCE,
            PelagoRelMdHetDistribution  .SOURCE,
            PelagoRelMdPacking          .SOURCE,
            PelagoRelMdRowCount         .SOURCE,
            PelagoRelMdExpressionLineage.SOURCE,
            PelagoRelMdSelfCost         .SOURCE(),
            DefaultRelMetadataProvider  .INSTANCE));
  }
}
