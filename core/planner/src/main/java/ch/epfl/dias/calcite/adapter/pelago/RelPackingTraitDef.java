/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to you under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.core.DeviceCross;
//import org.apache.calcite.rel.logical.LogicalDeviceCross;

/**
 * Definition of the device type trait.
 *
 * <p>Target device type is a physical property (i.e. a trait) because it can be
 * changed without loss of information. The converter to do this is the
 * {@link PelagoDeviceCross} operator.
 */
public class RelPackingTraitDef extends RelTraitDef<RelPacking> {
  public static final RelPackingTraitDef INSTANCE = new RelPackingTraitDef();

  protected RelPackingTraitDef() {}

  @Override public Class<RelPacking> getTraitClass() {
    return RelPacking.class;
  }

  @Override public String getSimpleName() {
    return "device";
  }

  @Override public RelNode convert(RelOptPlanner planner, RelNode rel, RelPacking to_packing,
                                   boolean allowInfiniteCostConverters) {
    if (rel.getConvention() != PelagoRel.CONVENTION()) return null;
    if (rel.getTraitSet().containsIfApplicable(to_packing)) return rel;

    RelNode p;
    if (to_packing == RelPacking.Packed) {
      p = PelagoPack  .create(rel, to_packing);
    } else {
      p = PelagoUnpack.create(rel, to_packing);
    }
    RelNode newRel = planner.register(p, rel);
    final RelTraitSet newTraitSet = rel.getTraitSet().replace(to_packing);
    if (!newRel.getTraitSet().equals(newTraitSet)) {
      newRel = planner.changeTraits(newRel, newTraitSet);
    }
    return newRel;
  }

  @Override public boolean canConvert(RelOptPlanner planner, RelPacking fromTrait, RelPacking toTrait) {
    return true;//fromTrait != toTrait;
  }

  @Override public RelPacking getDefault() {
    return RelPacking.UnPckd;
  }
}

// End RelDeviceTypeTraitDef.java
