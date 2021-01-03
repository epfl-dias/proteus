package ch.epfl.dias.calcite.adapter.pelago.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;

/**
 * Description of the distribution across device types of an input stream.
 */
public class RelHetDistribution implements PelagoTrait {
  public static final RelHetDistribution SPLIT        = new RelHetDistribution("hetSplit");
  public static final RelHetDistribution SPLIT_BRDCST = new RelHetDistribution("hetBrdcst");
  public static final RelHetDistribution SINGLETON    = new RelHetDistribution("hetSingle");

  protected final String distr;

  protected RelHetDistribution(String distr) {
    this.distr = distr;
  }

  public int getNumOfDeviceTypes(){
    return (this == SINGLETON) ? 1 : 2;
  }

  @Override public String toString() {
    return distr;
  }

  @Override public RelTraitDef getTraitDef() {
    return RelHetDistributionTraitDef.INSTANCE;
  }

  @Override public boolean satisfies(RelTrait trait) {
    return (trait == this);
  }

  @Override public void register(RelOptPlanner planner) {}
}
