package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.plan.RelMultipleTrait;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.util.ImmutableIntList;
import org.apache.calcite.util.Util;
import org.apache.calcite.util.mapping.Mapping;
import org.apache.calcite.util.mapping.Mappings;

import com.google.common.base.Preconditions;
import com.google.common.collect.Ordering;

import javax.annotation.Nonnull;

import java.util.List;
import java.util.Objects;

public class PelagoRelDistributions {
  public static final RelDistribution SPLIT = new RelDistributionSplit(RelDistributions.RANDOM_DISTRIBUTED, ImmutableIntList.of());
  public static final RelDistribution BRDCSPLIT = new RelDistributionSplit(RelDistributions.BROADCAST_DISTRIBUTED, ImmutableIntList.of());

  public static class RelDistributionSplit implements RelDistribution {
    private final RelDistribution type;
    private final ImmutableIntList keys;

    private RelDistributionSplit(RelDistribution type, ImmutableIntList keys) {
      this.type = type;
      this.keys = ImmutableIntList.copyOf(keys);
    }

    @Override public int hashCode() {
      return Objects.hash(type, keys);
    }

    @Override public boolean equals(Object obj) {
      return this == obj;
    }

    @Override public String toString() {
      return type.toString() + "_split";
    }

    @Override public void register(final RelOptPlanner planner) {
    }

    @Nonnull public Type getType() {
      return type.getType();
    }

    @Nonnull public List<Integer> getKeys() {
      return keys;
    }

    public RelDistributionTraitDef getTraitDef() {
      return RelDistributionTraitDef.INSTANCE;
    }

    public RelDistribution apply(Mappings.TargetMapping mapping) {
      return this;
    }

    public boolean satisfies(RelTrait trait) {
      return (trait == this || trait == RelDistributions.ANY);
    }

    @Override public boolean isTop() {
      return getType() == Type.ANY;
    }

    @Override public int compareTo(@Nonnull RelMultipleTrait o) {
      final RelDistribution distribution = (RelDistribution) o;
      return getType().compareTo(distribution.getType());
    }
  }
}
