package ch.epfl.dias.calcite.adapter.pelago;

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery;
import ch.epfl.dias.emitter.Binding;
import org.apache.calcite.plan.Convention;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.plan.RelOptPlanner;
import org.apache.calcite.plan.RelTraitSet;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.RelMetadataQuery;

import org.json4s.JsonAST;
import scala.Tuple2;

public interface PelagoRel extends RelNode {
    /** Calling convention for relational operations that occur in Pelago. */
    Convention CONVENTION = new Convention.Impl("Pelago", PelagoRel.class);

    public Tuple2<Binding, JsonAST.JValue> implement(RelDeviceType target);

    RelOptCost computeBaseSelfCost(RelOptPlanner planner, RelMetadataQuery mq);
}

