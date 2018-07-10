package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.plan.Context;
import org.apache.calcite.plan.RelOptCluster;
import org.apache.calcite.plan.RelOptSchema;
import org.apache.calcite.schema.SchemaPlus;
import org.apache.calcite.server.CalciteServerStatement;
import org.apache.calcite.tools.FrameworkConfig;
import org.apache.calcite.tools.Frameworks;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.tools.RelBuilderFactory;

public class PelagoRelBuilder extends RelBuilder {
  private PelagoRelBuilder(Context context, RelOptCluster cluster, RelOptSchema relOptSchema) {
    super(context, cluster, relOptSchema);
  }

  public static RelBuilder create(FrameworkConfig config) {
    final RelOptCluster[] clusters      = { null };
    final RelOptSchema [] relOptSchemas = { null };
    Frameworks.withPrepare(
      new Frameworks.PrepareAction<Void>(config) {
        public Void apply(RelOptCluster cluster, RelOptSchema relOptSchema,
            SchemaPlus rootSchema, CalciteServerStatement statement) {
          clusters[0] = cluster;
          relOptSchemas[0] = relOptSchema;
          return null;
        }
      });
    return new PelagoRelBuilder(config.getContext(), clusters[0], relOptSchemas[0]);
  }

  public static RelBuilderFactory proto(final Context context) {
    return new RelBuilderFactory() {
      public RelBuilder create(RelOptCluster cluster, RelOptSchema schema) {
        return new PelagoRelBuilder(context, cluster, schema);
      }
    };
  }
}
