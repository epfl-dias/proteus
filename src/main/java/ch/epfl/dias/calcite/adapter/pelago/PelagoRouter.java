//package ch.epfl.dias.calcite.adapter.pelago;
//
//import ch.epfl.dias.emitter.Binding;
//import ch.epfl.dias.emitter.PlanToJSON;
//import org.apache.calcite.plan.*;
//import org.apache.calcite.rel.RelDistribution;
//import org.apache.calcite.rel.RelDistributionTraitDef;
//import org.apache.calcite.rel.RelDistributions;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.convert.Converter;
//import org.apache.calcite.rel.convert.ConverterImpl;
//import org.apache.calcite.rel.core.Aggregate;
//import org.apache.calcite.rel.core.AggregateCall;
//import org.apache.calcite.rel.core.Exchange;
//import org.apache.calcite.rel.metadata.RelMetadataQuery;
//import org.apache.calcite.util.ImmutableBitSet;
//import org.json4s.JsonAST;
//import scala.Tuple2;
//
//import java.util.List;
//
//public class PelagoRouter extends Exchange implements PelagoRel, Converter {
//    protected RelTraitSet inTraits;
//
//    public PelagoRouter(RelOptCluster cluster, RelTraitSet traitSet, RelNode input, RelDistribution distribution){
//        super(cluster, traitSet, input, distribution);
//        assert getConvention() == PelagoRel.CONVENTION;
//        assert getConvention() == input.getConvention();
//        inTraits = input.getTraitSet();
//    }
//
//    @Override
//    public Exchange copy(RelTraitSet traitSet, RelNode input, RelDistribution distribution) {
//        return new PelagoRouter(getCluster(), traitSet, input, distribution);
//    }
//
//    @Override
//    public double estimateRowCount(RelMetadataQuery mq) {
//        double rc = super.estimateRowCount(mq);
//        if (getDistribution()           == RelDistributions.BROADCAST_DISTRIBUTED) rc = rc * 8;
//        return rc;
//    }
//
//    @Override public RelOptCost computeSelfCost(RelOptPlanner planner, RelMetadataQuery mq) {
//        RelOptCost base = super.computeSelfCost(planner, mq).multiplyBy(0.1);
//        if (getDistribution().getType() == RelDistribution.Type.HASH_DISTRIBUTED) base = base.multiplyBy(80);
//        return base;
//    }
//
//    @Override
//    public Tuple2<Binding, JsonAST.JValue> implement() {
//        return null;
//    }
//
//    @Override
//    public RelTraitSet getInputTraits() {
//        return inTraits;
//    }
//
//    @Override
//    public RelTraitDef getTraitDef() {
//        return RelDistributionTraitDef.INSTANCE;
//    }
//}
