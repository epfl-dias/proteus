package ch.epfl.dias.calcite.adapter.pelago;

import org.apache.calcite.adapter.enumerable.*;
import org.apache.calcite.linq4j.tree.Blocks;
import org.apache.calcite.linq4j.tree.Expressions;
import org.apache.calcite.linq4j.tree.Primitive;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;

import java.util.List;

/**
 * Relational expression representing a scan of a Pelago file.
 *
 * Based on:
 * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTableScan.java
 *
 * <p>Like any table scan, it serves as a leaf node of a query tree.</p>
 */
public class PelagoTableScan extends TableScan implements EnumerableRel {
  final PelagoTable pelagoTable;
  final int[] fields;

  protected PelagoTableScan(RelOptCluster cluster, RelOptTable table,
                            PelagoTable pelagoTable, int[] fields) {
    super(cluster, cluster.traitSetOf(EnumerableConvention.INSTANCE), table);
    this.pelagoTable = pelagoTable;
    this.fields = fields;

    assert pelagoTable != null;
  }

  @Override public RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
    assert inputs.isEmpty();
    return new PelagoTableScan(getCluster(), table, pelagoTable, fields);
  }

  @Override public RelWriter explainTerms(RelWriter pw) {
    return super.explainTerms(pw)
        .item("fields", Primitive.asList(fields));
  }

  @Override public RelDataType deriveRowType() {
    final List<RelDataTypeField> fieldList = table.getRowType().getFieldList();
    final RelDataTypeFactory.Builder builder =
        getCluster().getTypeFactory().builder();
    for (int field : fields) {
      builder.add(fieldList.get(field));
    }
    return builder.build();
  }

  @Override public void register(RelOptPlanner planner) {
    planner.addRule(PelagoProjectTableScanRule.INSTANCE);
  }

  @Override public RelOptCost computeSelfCost(RelOptPlanner planner,
      RelMetadataQuery mq) {
    // Multiply the cost by a factor that makes a scan more attractive if it
    // has significantly fewer fields than the original scan.
    //
    // The "+ 2D" on top and bottom keeps the function fairly smooth.
    //
    // For example, if table has 3 fields, project has 1 field,
    // then factor = (1 + 2) / (3 + 2) = 0.6
    return super.computeSelfCost(planner, mq)
        .multiplyBy(((double) fields.length + 2D)
            / ((double) table.getRowType().getFieldCount() + 2D));
  }

  public Result implement(EnumerableRelImplementor implementor, Prefer pref) {
    PhysType physType =
        PhysTypeImpl.of(
            implementor.getTypeFactory(),
            getRowType(),
            pref.preferArray());

    return implementor.result(
        physType,
        Blocks.toBlock(
            Expressions.call(table.getExpression(PelagoTable.class),
                "project", implementor.getRootExpression(),
                Expressions.constant(fields))));
  }

  public String getPelagoRelName(){
      return pelagoTable.getPelagoRelName();
  }
}

// End CsvTableScan.java
