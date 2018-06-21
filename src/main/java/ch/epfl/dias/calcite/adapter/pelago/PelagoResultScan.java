package ch.epfl.dias.calcite.adapter.pelago;

import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoRules;
import org.apache.calcite.DataContext;
//import org.apache.calcite.adapter.csv.CsvTranslatableTable;
//import org.apache.calcite.adapter.csv.JsonTable;
import org.apache.calcite.adapter.enumerable.*;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.interpreter.Row;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Queryable;
import org.apache.calcite.linq4j.function.Function1;
import org.apache.calcite.linq4j.tree.*;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.schema.*;
import org.apache.calcite.util.BuiltInMethod;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;


/**
 * Relational expression representing a scan of a Pelago file.
 *
 * Based on:
 * https://github.com/apache/calcite/blob/master/example/csv/src/main/java/org/apache/calcite/adapter/csv/CsvTableScan.java
 *
 * <p>Like any table scan, it serves as a leaf node of a query tree.</p>
 */
public class PelagoResultScan extends TableScan implements EnumerableRel {
  final PelagoResultTable pelagoTable;
  final int[] fields;
  final Class elementType = Object[].class;

  protected PelagoResultScan(RelOptCluster cluster, RelOptTable table,
                             PelagoResultTable pelagoTable, int[] fields) {
    super(cluster, cluster.traitSet().plus(PelagoRel.CONVENTION).plus(RelDistributions.RANDOM_DISTRIBUTED), table);
    this.pelagoTable = pelagoTable;
    this.fields = fields;

    assert pelagoTable != null;
  }

  @Override public RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
    assert inputs.isEmpty();
//    return new PelagoResultScan(getCluster(), table, pelagoTable, fields);
    return new PelagoResultScan(getCluster(), null, pelagoTable, fields);
  }

//  @Override public RelWriter explainTerms(RelWriter pw) {
//    return super.explainTerms(pw)
//        .item("fields", Primitive.asList(fields));
//  }
//
//  @Override public RelDataType deriveRowType() {
//    final List<RelDataTypeField> fieldList = table.getRowType().getFieldList();
//    final RelDataTypeFactory.Builder builder =
//        getCluster().getTypeFactory().builder();
//    for (int field : fields) {
//      builder.add(fieldList.get(field));
//    }
//    return builder.build();
//  }

//  @Override public void register(RelOptPlanner planner) {
//    for (RelOptRule rule : PelagoRules.RULES) planner.addRule(rule);
//    planner.addRule(PelagoProjectTableScanRule.INSTANCE);
//  }
//
  @Override public RelOptCost computeSelfCost(RelOptPlanner planner,
      RelMetadataQuery mq) {
    // Multiply the cost by a factor that makes a scan more attractive if it
    // has significantly fewer fields than the original scan.
    //
    // The "+ 2D" on top and bottom keeps the function fairly smooth.
    //
    // For example, if table has 3 fields, project has 1 field,
    // then factor = (1 + 2) / (3 + 2) = 0.6
//    return super.computeSelfCost(planner, mq)
//        .multiplyBy(((double) fields.length + 2D)
//            / ((double) table.getRowType().getFieldCount() + 2D));
    return planner.getCostFactory().makeTinyCost();
  }
  public Result implement(EnumerableRelImplementor implementor, Prefer pref) {
    // Note that representation is ARRAY. This assumes that the table
    // returns a Object[] for each record. Actually a Table<T> can
    // return any type T. And, if it is a JdbcTable, we'd like to be
    // able to generate alternate accessors that return e.g. synthetic
    // records {T0 f0; T1 f1; ...} and don't box every primitive value.
    final PhysType physType =
            PhysTypeImpl.of(
                    implementor.getTypeFactory(),
                    getRowType(),
                    format());
    final Expression expression = getExpression(physType);
    return implementor.result(physType, Blocks.toBlock(expression));
  }

  private Expression getExpression(PhysType physType) {
    final Expression expression = table.getExpression(ScannableTable.class);
    final Expression expression2 = toEnumerable(expression);
    assert Types.isAssignableFrom(Enumerable.class, expression2.getType());
    return toRows(physType, expression2);
  }




  private Expression toEnumerable(Expression expression) {
    final Type type = expression.getType();
    if (Types.isArray(type)) {
      if (Types.toClass(type).getComponentType().isPrimitive()) {
        expression =
                Expressions.call(BuiltInMethod.AS_LIST.method, expression);
      }
      return Expressions.call(BuiltInMethod.AS_ENUMERABLE.method, expression);
    } else if (Types.isAssignableFrom(Iterable.class, type)
            && !Types.isAssignableFrom(Enumerable.class, type)) {
      return Expressions.call(BuiltInMethod.AS_ENUMERABLE2.method,
              expression);
    } else if (Types.isAssignableFrom(Queryable.class, type)) {
      // Queryable extends Enumerable, but it's too "clever", so we call
      // Queryable.asEnumerable so that operations such as take(int) will be
      // evaluated directly.
      return Expressions.call(expression,
              BuiltInMethod.QUERYABLE_AS_ENUMERABLE.method);
    }
    return expression;
  }

  private Expression toRows(PhysType physType, Expression expression) {
    if (physType.getFormat() == JavaRowFormat.SCALAR
            && Object[].class.isAssignableFrom(elementType)
            && getRowType().getFieldCount() == 1
            && (table.unwrap(ScannableTable.class) != null
            || table.unwrap(FilterableTable.class) != null
            || table.unwrap(ProjectableFilterableTable.class) != null)) {
      return Expressions.call(BuiltInMethod.SLICE0.method, expression);
    }
    JavaRowFormat oldFormat = format();
    if (physType.getFormat() == oldFormat && !hasCollectionField(rowType)) {
      return expression;
    }
    final ParameterExpression row_ =
            Expressions.parameter(elementType, "row");
    final int fieldCount = table.getRowType().getFieldCount();
    List<Expression> expressionList = new ArrayList<>(fieldCount);
    for (int i = 0; i < fieldCount; i++) {
      expressionList.add(fieldExpression(row_, i, physType, oldFormat));
    }
    return Expressions.call(expression,
            BuiltInMethod.SELECT.method,
            Expressions.lambda(Function1.class, physType.record(expressionList),
                    row_));
  }

  private Expression fieldExpression(ParameterExpression row_, int i,
                                     PhysType physType, JavaRowFormat format) {
    final Expression e =
            format.field(row_, i, null, physType.getJavaFieldType(i));
    final RelDataType relFieldType =
            physType.getRowType().getFieldList().get(i).getType();
    switch (relFieldType.getSqlTypeName()) {
      case ARRAY:
      case MULTISET:
        // We can't represent a multiset or array as a List<Employee>, because
        // the consumer does not know the element type.
        // The standard element type is List.
        // We need to convert to a List<List>.
        final JavaTypeFactory typeFactory =
                (JavaTypeFactory) getCluster().getTypeFactory();
        final PhysType elementPhysType = PhysTypeImpl.of(
                typeFactory, relFieldType.getComponentType(), JavaRowFormat.CUSTOM);
        final MethodCallExpression e2 =
                Expressions.call(BuiltInMethod.AS_ENUMERABLE2.method, e);
        final RelDataType dummyType = this.rowType;
        final Expression e3 =
                elementPhysType.convertTo(e2,
                        PhysTypeImpl.of(typeFactory, dummyType, JavaRowFormat.LIST));
        return Expressions.call(e3, BuiltInMethod.ENUMERABLE_TO_LIST.method);
      default:
        return e;
    }
  }

  private JavaRowFormat format() {
    int fieldCount = getRowType().getFieldCount();
    if (fieldCount == 0) {
      return JavaRowFormat.LIST;
    }
    if (Object[].class.isAssignableFrom(elementType)) {
      return fieldCount == 1 ? JavaRowFormat.SCALAR : JavaRowFormat.ARRAY;
    }
    if (Row.class.isAssignableFrom(elementType)) {
      return JavaRowFormat.ROW;
    }
    if (fieldCount == 1 && (Object.class == elementType
            || Primitive.is(elementType)
            || Number.class.isAssignableFrom(elementType))) {
      return JavaRowFormat.SCALAR;
    }
    return JavaRowFormat.CUSTOM;
  }

  private boolean hasCollectionField(RelDataType rowType) {
    for (RelDataTypeField field : rowType.getFieldList()) {
      switch (field.getType().getSqlTypeName()) {
        case ARRAY:
        case MULTISET:
          return true;
      }
    }
    return false;
  }
}

// End CsvTableScan.java
