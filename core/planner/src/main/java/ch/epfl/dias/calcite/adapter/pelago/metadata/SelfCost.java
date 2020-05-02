package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.linq4j.tree.Types;
import org.apache.calcite.plan.RelOptCost;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.Metadata;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.RelMetadataQuery;

import ch.epfl.dias.calcite.adapter.pelago.RelPacking;

import java.lang.reflect.Method;
//
//public interface SelfCost extends Metadata {
//  public static final Method method = Types.lookupMethod(SelfCost.class, "selfCost");
//  MetadataDef<SelfCost> DEF = MetadataDef.of(SelfCost.class, SelfCost.Handler.class, method);
//
//  /** Determines how the self cost of a RelNode. */
//  RelOptCost selfCost();
//
//  /** Handler API. */
//  interface Handler extends MetadataHandler<SelfCost> {
//    RelOptCost selfCost(RelNode r, RelMetadataQuery mq);
//  }
//}
