package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.JaninoRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;

import com.google.common.base.Preconditions;

import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

public class PelagoRelMetadataQuery extends RelMetadataQuery {
  private DeviceType.Handler      deviceTypeHandler;
  private Packing.Handler         packingHandler;
//  private SelfCost.Handler        selfCostHandler;
  private HetDistribution.Handler hetDistrHandler;
  private ComputeDevice.Handler   computeTypeHandler;

  protected PelagoRelMetadataQuery(JaninoRelMetadataProvider metadataProvider,
                                   RelMetadataQuery prototype) {
    super(metadataProvider, prototype);
    deviceTypeHandler  = initialHandler(DeviceType.Handler.class);
    packingHandler     = initialHandler(Packing   .Handler.class);
//    selfCostHandler    = initialHandler(SelfCost  .Handler.class);
    hetDistrHandler    = initialHandler(HetDistribution.Handler.class);
    computeTypeHandler = initialHandler(ComputeDevice  .Handler.class);
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
   * {@link DeviceType#deviceType()}
   * statistic.
   *
   * @param rel         the relational expression
   */
  public RelHetDistribution hetDistribution(RelNode rel) {
    for (;;) {
      try {
        return hetDistrHandler.hetDistribution(rel, this);
      } catch (JaninoRelMetadataProvider.NoHandler e) {
        hetDistrHandler = revise(e.relClass, HetDistribution.DEF);
      }
    }
  }

  /**
   * Returns the
   * {@link DeviceType#deviceType()}
   * statistic.
   *
   * @param rel         the relational expression
   */
  public RelComputeDevice computeType(RelNode rel) {
    for (;;) {
      try {
        return computeTypeHandler.computeType(rel, this);
      } catch (JaninoRelMetadataProvider.NoHandler e) {
        computeTypeHandler = revise(e.relClass, ComputeDevice.DEF);
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

//  /**
//   * Returns the
//   * {@link SelfCost#selfCost()}
//   * .
//   *
//   * @param rel         the relational expression
//   */
//  public RelOptCost selfCost(RelNode rel) {
//    for (;;) {
//      try {
//        return selfCostHandler.(rel, this);
//      } catch (JaninoRelMetadataProvider.NoHandler e) {
//        selfCostHandler = revise(e.relClass, SelfCost.DEF);
//      }
//    }
//  }
}
