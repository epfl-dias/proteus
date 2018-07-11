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
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.SingleRel;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Exchange;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.SetOp;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexProgram;

import com.google.common.collect.ImmutableList;

import ch.epfl.dias.calcite.adapter.pelago.PelagoDeviceCross;
import ch.epfl.dias.calcite.adapter.pelago.PelagoPack;
import ch.epfl.dias.calcite.adapter.pelago.PelagoTable;
import ch.epfl.dias.calcite.adapter.pelago.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

import java.util.List;

//import org.apache.calcite.rel.RelDeviceType;
//import org.apache.calcite.rel.core.DeviceCross;

/**
 * RelMdCollation supplies a default implementation of
 * {@link PelagoRelMetadataQuery#packing}
 * for the standard logical algebra.
 */
public class PelagoRelMdPacking
    implements MetadataHandler<Packing> {
  public static final RelMetadataProvider SOURCE =
      ReflectiveRelMetadataProvider.reflectiveSource(
          Packing.method, new PelagoRelMdPacking());

  //~ Constructors -----------------------------------------------------------

  protected PelagoRelMdPacking() {}

  //~ Methods ----------------------------------------------------------------

  public MetadataDef<Packing> getDef() {
    return Packing.DEF;
  }

  /** Fallback method to deduce deviceType for any relational expression not
   * handled by a more specific method.
   *
   * @param rel Relational expression
   * @return Relational expression's deviceType
   */
  public RelPacking packing(RelNode rel, RelMetadataQuery mq) {
    return RelPacking.UnPckd;
  }

  public RelPacking packing(SingleRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).packing(rel.getInput());
  }

  public RelPacking packing(BiRel rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).packing(rel.getLeft());
  }

  public RelPacking packing(SetOp rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).packing(rel.getInputs().get(0));
  }

  public RelPacking packing(TableScan scan, RelMetadataQuery mq) {
    return table(scan.getTable());
  }

  public RelPacking packing(Project project, RelMetadataQuery mq) {
    return project(mq, project.getInput(), project.getProjects());
  }

  public RelPacking packing(Values values, RelMetadataQuery mq) {
    return values(values.getRowType(), values.getTuples());
  }

  public RelPacking packing(PelagoDeviceCross devicecross, RelMetadataQuery mq) {
    return devicecross(mq, devicecross.getInput());
  }

  public RelPacking packing(HepRelVertex rel, RelMetadataQuery mq) {
    return ((PelagoRelMetadataQuery) mq).packing(rel.getCurrentRel());
  }

  public RelPacking packing(PelagoPack rel, RelMetadataQuery mq) {
    return pack();
  }

  public RelPacking packing(PelagoUnpack rel, RelMetadataQuery mq) {
    return unpack();
  }

  // Helper methods

  /** Helper method to determine a
   * {@link TableScan}'s deviceType. */
  public static RelPacking table(RelOptTable table) {
    return RelPacking.UnPckd;
  }

  /** Helper method to determine a
   * {@link TableScan}'s deviceType. */
  public static RelPacking table(PelagoTable table) {
    return table.getPacking();
  }

  /** Helper method to determine a
   * {@link Sort}'s deviceType. */
  public static RelPacking sort(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).packing(input);
  }

  /** Helper method to determine a
   * {@link Filter}'s deviceType. */
  public static RelPacking filter(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).packing(input);
  }

  /** Helper method to determine a
   * {@link Aggregate}'s deviceType. */
  public static RelPacking aggregate(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).packing(input);
  }

  /** Helper method to determine a
   * {@link Exchange}'s deviceType. */
  public static RelPacking exchange(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).packing(input);
  }

  /** Helper method to determine a
   * limit's deviceType. */
  public static RelPacking limit(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).packing(input);
  }

  /** Helper method to determine a
   * {@link org.apache.calcite.rel.core.Calc}'s deviceType. */
  public static RelPacking calc(RelMetadataQuery mq, RelNode input,
      RexProgram program) {
    throw new AssertionError(); // TODO:
  }

  /** Helper method to determine a {@link Project}'s collation. */
  public static RelPacking project(RelMetadataQuery mq, RelNode input,
      List<? extends RexNode> projects) {
    final RelPacking inputdeviceType = ((PelagoRelMetadataQuery) mq).packing(input);
//    final Mappings.TargetMapping mapping =
//        Project.getPartialMapping(input.getRowType().getFieldCount(),
//            projects);
    return inputdeviceType; //.apply(mapping); // TODO: Should we do something here ?
  }

  /** Helper method to determine a
   * {@link Values}'s deviceType. */
  public static RelPacking values(RelDataType rowType,
      ImmutableList<ImmutableList<RexLiteral>> tuples) {
    return RelPacking.UnPckd;
  }

  /** Helper method to determine an
   * {@link Exchange}'s
   * or {@link org.apache.calcite.rel.core.SortExchange}'s deviceType. */
  public static RelPacking devicecross(RelMetadataQuery mq, RelNode input) {
    return ((PelagoRelMetadataQuery) mq).packing(input);
  }

  public static RelPacking unpack() {
    return RelPacking.UnPckd;
  }

  public static RelPacking pack() {
    return RelPacking.Packed;
  }
}

// End RelMdDeviceType.java
