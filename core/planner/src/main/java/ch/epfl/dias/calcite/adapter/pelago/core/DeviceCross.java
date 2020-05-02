//package ch.epfl.dias.calcite.adapter.pelago.core;
//
//import org.apache.calcite.plan.RelOptCluster;
//import org.apache.calcite.plan.RelOptCost;
//import org.apache.calcite.plan.RelOptPlanner;
//import org.apache.calcite.plan.RelTraitSet;
//import org.apache.calcite.rel.RelDeviceType;
//import org.apache.calcite.rel.RelDeviceTypeTraitDef;
//import org.apache.calcite.rel.RelInput;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.RelWriter;
//import org.apache.calcite.rel.SingleRel;
//import org.apache.calcite.rel.metadata.RelMetadataQuery;
//import org.apache.calcite.util.Util;
//
//import com.google.common.base.Preconditions;
//
//import java.util.List;
//
///**
// * Relational expression that imposes a particular device target on its input
// * without otherwise changing its content.
// */
//public abstract class DeviceCross extends SingleRel {
//  //~ Instance fields --------------------------------------------------------
//
//  public final RelDeviceType deviceType;
//
//  //~ Constructors -----------------------------------------------------------
//
//  /**
//   * Creates an Exchange.
//   *
//   * @param cluster   Cluster this relational expression belongs to
//   * @param traitSet  Trait set
//   * @param input     Input relational expression
//   * @param deviceType Distribution specification
//   */
//  protected DeviceCross(RelOptCluster cluster, RelTraitSet traitSet, RelNode input,
//      RelDeviceType deviceType) {
//    super(cluster, traitSet, input);
//    this.deviceType = Preconditions.checkNotNull(deviceType);
//
//    assert traitSet.containsIfApplicable(deviceType)
//        : "traits=" + traitSet + ", device" + deviceType;
//    assert deviceType != RelDeviceType.ANY;
//  }
//
//  /**
//   * Creates a Exchange by parsing serialized output.
//   */
//  public DeviceCross(RelInput input) {
//    this(input.getCluster(), input.getTraitSet().plus(input.getCollation()),
//        input.getInput(),
//        RelDeviceTypeTraitDef.INSTANCE.canonize(input.getDeviceType()));
//  }
//
//  //~ Methods ----------------------------------------------------------------
//
//  @Override public final DeviceCross copy(RelTraitSet traitSet,
//      List<RelNode> inputs) {
//    return copy(traitSet, sole(inputs), deviceType);
//  }
//
//  public abstract DeviceCross copy(RelTraitSet traitSet, RelNode newInput,
//      RelDeviceType deviceType);
//
//  /** Returns the distribution of the rows returned by this Exchange. */
//  public RelDeviceType getDeviceType() {
//    return deviceType;
//  }
//
//  @Override public RelOptCost computeSelfCost(RelOptPlanner planner,
//      RelMetadataQuery mq) {
//    // Higher cost if rows are wider discourages pushing a project through an
//    // exchange.
//    double rowCount = mq.getRowCount(this);
//    double bytesPerRow = getRowType().getFieldCount() * 4;
//    return planner.getCostFactory().makeCost(
//        Util.nLogN(rowCount) * bytesPerRow, rowCount, 0);
//  }
//
//  public RelWriter explainTerms(RelWriter pw) {
//    return super.explainTerms(pw)
//        .item("device", deviceType);
//  }
//}
//
//// End DeviceCross.java
