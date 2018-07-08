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
package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.plan.RelOptTable;
import org.apache.calcite.plan.hep.HepRelVertex;
import org.apache.calcite.rel.BiRel;
//import org.apache.calcite.rel.RelDeviceType;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.Aggregate;
//import org.apache.calcite.rel.core.DeviceCross;
import org.apache.calcite.rel.core.Exchange;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.SetOp;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.metadata.BuiltInMetadata;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexProgram;
import org.apache.calcite.util.BuiltInMethod;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTable;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

import java.util.List;

/**
 * RelMdCollation supplies a default implementation of
 * {@link RelMetadataQuery#deviceType}
 * for the standard logical algebra.
 */
public class RelMdDeviceType
    implements MetadataHandler<DeviceType> {
  public static final RelMetadataProvider SOURCE =
      ReflectiveRelMetadataProvider.reflectiveSource(
          DeviceType.method, new RelMdDeviceType());

  //~ Constructors -----------------------------------------------------------

  protected RelMdDeviceType() {}

  //~ Methods ----------------------------------------------------------------

  public MetadataDef<DeviceType> getDef() {
    return DeviceType.DEF;
  }

  /** Fallback method to deduce deviceType for any relational expression not
   * handled by a more specific method.
   *
   * @param rel Relational expression
   * @return Relational expression's deviceType
   */
  public RelDeviceType deviceType(RelNode rel, RelMetadataQuery mq) {
    return RelDeviceType.X86_64;
  }

  public RelDeviceType deviceType(SingleRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getInput());
  }

  public RelDeviceType deviceType(BiRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getLeft());
  }

  public RelDeviceType deviceType(SetOp rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getInputs().get(0));
  }

  public RelDeviceType deviceType(TableScan scan, RelMetadataQuery mq) {
    return table(scan.getTable());
  }

  public RelDeviceType deviceType(Project project, RelMetadataQuery mq) {
    return project(mq, project.getInput(), project.getProjects());
  }

  public RelDeviceType deviceType(Values values, RelMetadataQuery mq) {
    return values(values.getRowType(), values.getTuples());
  }

  public RelDeviceType deviceType(PelagoDeviceCross devicecross, RelMetadataQuery mq) {
    return devicecross(devicecross.getDeviceType());
  }

  public RelDeviceType deviceType(HepRelVertex rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).deviceType(rel.getCurrentRel());
  }

  // Helper methods

  /** Helper method to determine a
   * {@link TableScan}'s deviceType. */
  public static RelDeviceType table(RelOptTable table) {
    return RelDeviceType.X86_64;
  }

  /** Helper method to determine a
   * {@link TableScan}'s deviceType. */
  public static RelDeviceType table(PelagoTable table) {
    return table.getDeviceType();
  }

  /** Helper method to determine a
   * {@link Sort}'s deviceType. */
  public static RelDeviceType sort(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).deviceType(input);
  }

  /** Helper method to determine a
   * {@link Filter}'s deviceType. */
  public static RelDeviceType filter(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).deviceType(input);
  }

  /** Helper method to determine a
   * {@link Aggregate}'s deviceType. */
  public static RelDeviceType aggregate(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).deviceType(input);
  }

  /** Helper method to determine a
   * {@link Exchange}'s deviceType. */
  public static RelDeviceType exchange(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).deviceType(input);
  }

  /** Helper method to determine a
   * limit's deviceType. */
  public static RelDeviceType limit(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).deviceType(input);
  }

  /** Helper method to determine a
   * {@link org.apache.calcite.rel.core.Calc}'s deviceType. */
  public static RelDeviceType calc(RelMetadataQuery mq, RelNode input,
      RexProgram program) {
    throw new AssertionError(); // TODO:
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

  /** Helper method to determine a
   * {@link Values}'s deviceType. */
  public static RelDeviceType values(RelDataType rowType,
      ImmutableList<ImmutableList<RexLiteral>> tuples) {
    return RelDeviceType.ANY;
  }

  /** Helper method to determine an
   * {@link Exchange}'s
   * or {@link org.apache.calcite.rel.core.SortExchange}'s deviceType. */
  public static RelDeviceType devicecross(RelDeviceType deviceType) {
    return deviceType;
  }
}

// End RelMdDeviceType.java
