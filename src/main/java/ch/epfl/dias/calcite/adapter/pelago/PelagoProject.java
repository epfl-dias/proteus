//package ch.epfl.dias.calcite.adapter.pelago;
//
//import ch.epfl.dias.emitter.Binding;
//import ch.epfl.dias.emitter.PlanToJSON;
//import com.google.common.base.Supplier;
//import org.apache.calcite.adapter.java.JavaTypeFactory;
//import org.apache.calcite.plan.RelOptCluster;
//import org.apache.calcite.plan.RelOptCost;
//import org.apache.calcite.plan.RelOptPlanner;
//import org.apache.calcite.plan.RelTraitSet;
//import org.apache.calcite.rel.RelDistribution;
//import org.apache.calcite.rel.RelDistributionTraitDef;
//import org.apache.calcite.rel.RelNode;
//import org.apache.calcite.rel.core.Project;
//import org.apache.calcite.rel.metadata.RelMdDistribution;
//import org.apache.calcite.rel.metadata.RelMetadataQuery;
//import org.apache.calcite.rel.type.RelDataType;
//import org.apache.calcite.rex.RexNode;
//import org.json4s.JsonAST;
//import scala.Tuple2;
//
//import java.util.List;
//
///**
// * Implementation of {@link org.apache.calcite.rel.core.Project}
// * relational expression in Pelago.
// */
//public class PelagoProject extends Project implements PelagoRel {
//    public PelagoProject(RelOptCluster cluster, RelTraitSet traitSet,
//                         RelNode input, List<? extends RexNode> projects, RelDataType rowType) {
//        super(cluster, input.getTraitSet().replace(PelagoRel.CONVENTION).replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
//              public RelDistribution get() { return RelMdDistribution.project(cluster.getMetadataQuery(), input, projects); }
//        }), input, projects, rowType);
//        assert getConvention() == PelagoRel.CONVENTION;
////        assert getConvention() == input.getConvention();
//    }
//
//    @Override public Project copy(RelTraitSet traitSet, RelNode input,
//                                  List<RexNode> projects, RelDataType rowType) {
//        return new PelagoProject(getCluster(), traitSet, input, projects,
//                rowType);
//    }
//
//    @Override public RelOptCost computeSelfCost(RelOptPlanner planner,
//                                                RelMetadataQuery mq) {
//        return super.computeSelfCost(planner, mq).multiplyBy(0.001); //almost 0 cost in Pelago
//    }
//
//    @Override
//    public Tuple2<Binding, JsonAST.JValue> implement() {
//        return null;
//    }
//}
