package ch.epfl.dias.calcite.adapter.pelago.metadata;

import org.apache.calcite.linq4j.tree.Types;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.metadata.Metadata;
import org.apache.calcite.rel.metadata.MetadataDef;
import org.apache.calcite.rel.metadata.MetadataHandler;
import org.apache.calcite.rel.metadata.RelMetadataQuery;

import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelHetDistribution;

import java.lang.reflect.Method;

/** Metadata about how a relational expression is distributed to heterogeneous hardware.
 *
 * <p>If you are an operator consuming a relational expression, which subset
 * of the rows are you seeing? You might be seeing all of them (BROADCAST
 * or SINGLETON), only those whose key column values have a particular hash
 * code (HASH) or only those whose column values have particular values or
 * ranges of values (RANGE).
 *
 * <p>When a relational expression is partitioned, it is often partitioned
 * among nodes, but it may be partitioned among threads running on the same
 * node. */
public interface HetDistribution extends Metadata {
  public static final Method method = Types.lookupMethod(HetDistribution.class, "hetDistribution");
  MetadataDef<HetDistribution> DEF = MetadataDef.of(HetDistribution.class,
      HetDistribution.Handler.class, method);

  /** Determines how the rows are distributed across devices. */
  RelHetDistribution hetDistribution();

  /** Handler API. */
  interface Handler extends MetadataHandler<HetDistribution> {
    RelHetDistribution hetDistribution(RelNode r, RelMetadataQuery mq);
  }
}
