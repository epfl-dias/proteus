package ch.epfl.dias.calcite.adapter.pelago.traits;

import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTrait;
import org.apache.calcite.plan.RelTraitDef;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;

import java.util.HashMap;
import java.util.Map;

/**
 * Description of the distribution across homogeneous devices, of an input stream.
 */
public class RelHomDistribution implements PelagoTrait {
  protected static final Map<RelDistribution, RelHomDistribution> available_distributions = new HashMap<>();

  public static final RelHomDistribution RANDOM = new RelHomDistribution("homRandom", RelDistributions.RANDOM_DISTRIBUTED);
  public static final RelHomDistribution BRDCST = new RelHomDistribution("homBrdcst", RelDistributions.BROADCAST_DISTRIBUTED);
  public static final RelHomDistribution SINGLE = new RelHomDistribution("homSingle", RelDistributions.SINGLETON);

  protected final String str;
  protected final RelDistribution distribution;

  protected RelHomDistribution(String str, RelDistribution distribution) {
    this.str = str;
    this.distribution = distribution;

    //Check that we do not already have a distribution with the same RelDistribution
    assert(!available_distributions.containsKey(distribution));

    available_distributions.put(distribution, this);
  }

  @Override public String toString() {
    return str;
  }

  @Override public RelTraitDef getTraitDef() {
    return RelHomDistributionTraitDef.INSTANCE;
  }

  public RelDistribution getDistribution(){
    return distribution;
  }

  public static RelHomDistribution from(RelDistribution distr){
    return available_distributions.get(distr);
  }

  @Override public boolean satisfies(RelTrait trait) {
    return (trait == this);
  }

  @Override public void register(RelOptPlanner planner) {}
}
