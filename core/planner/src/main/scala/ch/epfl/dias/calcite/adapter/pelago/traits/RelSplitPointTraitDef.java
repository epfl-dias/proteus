package ch.epfl.dias.calcite.adapter.pelago.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.rel.RelNode;

public class RelSplitPointTraitDef extends RelTraitDef<RelSplitPoint> {
  public static final RelSplitPointTraitDef INSTANCE = new RelSplitPointTraitDef();

  protected RelSplitPointTraitDef() {}

  @Override public Class<RelSplitPoint> getTraitClass() {
    return RelSplitPoint.class;
  }

  @Override public String getSimpleName() {
    return "split";
  }

  @Override public RelNode convert(RelOptPlanner planner, RelNode rel, RelSplitPoint toDevice,
                                   boolean allowInfiniteCostConverters) {
    if (!rel.getTraitSet().containsIfApplicable(toDevice)) return null;
    return rel;
  }

  @Override public boolean canConvert(RelOptPlanner planner, RelSplitPoint fromTrait,
      RelSplitPoint toDevice) {
    return fromTrait == RelSplitPoint.NONE() || toDevice == RelSplitPoint.NONE();
  }

  @Override public RelSplitPoint getDefault() {
    return RelSplitPoint.NONE();
  }
}
