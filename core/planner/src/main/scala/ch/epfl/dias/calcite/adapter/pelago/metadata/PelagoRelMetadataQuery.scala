/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2017
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
 */

package ch.epfl.dias.calcite.adapter.pelago.metadata

import ch.epfl.dias.calcite.adapter.pelago.traits._
import org.apache.calcite.rel.{RelDistribution, RelNode}
import org.apache.calcite.rel.metadata.RelMetadataQueryBase.initialHandler
import org.apache.calcite.rel.metadata.{
  ChainedRelMetadataProvider,
  JaninoRelMetadataProvider,
  RelMetadataQuery,
  RelMetadataQueryBase
}

import java.util

object PelagoRelMetadataQuery {
  def instance: PelagoRelMetadataQuery = {
    RelMetadataQueryBase.THREAD_PROVIDERS.set(
      JaninoRelMetadataProvider.of(
        ChainedRelMetadataProvider.of(
          util.List.of(
            RelMetadataQueryBase.THREAD_PROVIDERS.get,
            PelagoRelMetadataProvider.INSTANCE
          )
        )
      )
    )
    new PelagoRelMetadataQuery
  }
}

class PelagoRelMetadataQuery protected () extends RelMetadataQuery {
  private var deviceTypeHandler = initialHandler(classOf[DeviceType.Handler])
  private var packingHandler = initialHandler(classOf[Packing.Handler])
  //    selfCostHandler    = initialHandler(SelfCost  .Handler.class);
  private var hetDistrHandler = initialHandler(classOf[HetDistribution.Handler])
  private var homDistrHandler = initialHandler(classOf[HomDistribution.Handler])
  private var computeTypeHandler = initialHandler(
    classOf[ComputeDevice.Handler]
  )

  def deviceType(rel: RelNode): RelDeviceType = {
    while (true)
      try return deviceTypeHandler.deviceType(rel, this)
      catch {
        case e: JaninoRelMetadataProvider.NoHandler =>
          deviceTypeHandler = revise(e.relClass, DeviceType.DEF)
      }
    null
  }

  def hetDistribution(rel: RelNode): RelHetDistribution = {
    while (true)
      try return hetDistrHandler.hetDistribution(rel, this)
      catch {
        case e: JaninoRelMetadataProvider.NoHandler =>
          hetDistrHandler = revise(e.relClass, HetDistribution.DEF)
      }
    null
  }

  def homDistribution(rel: RelNode): RelHomDistribution = {
    while (true)
      try return homDistrHandler.homDistribution(rel, this)
      catch {
        case e: JaninoRelMetadataProvider.NoHandler =>
          homDistrHandler = revise(e.relClass, HomDistribution.DEF)
      }
    null
  }

  override def distribution(rel: RelNode): RelDistribution = {
    assert(false)
    null
  }

  def computeType(rel: RelNode): RelComputeDevice = {
    while (true)
      try return computeTypeHandler.computeType(rel, this)
      catch {
        case e: JaninoRelMetadataProvider.NoHandler =>
          computeTypeHandler = revise(e.relClass, ComputeDevice.DEF)
      }
    null
  }

  def packing(rel: RelNode): RelPacking = {
    while (true)
      try return packingHandler.packing(rel, this)
      catch {
        case e: JaninoRelMetadataProvider.NoHandler =>
          packingHandler = revise(e.relClass, Packing.DEF)
      }
    null
  }
}
