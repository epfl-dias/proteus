package ch.epfl.dias.calcite.adapter.pelago;

import com.google.common.base.Supplier;
import com.google.common.collect.ImmutableList;
import org.apache.calcite.DataContext;
import org.apache.calcite.adapter.enumerable.*;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.interpreter.Source;
import org.apache.calcite.linq4j.*;
import org.apache.calcite.linq4j.tree.*;
import org.apache.calcite.materialize.MaterializationService;
import org.apache.calcite.plan.*;
import org.apache.calcite.prepare.RelOptTableImpl;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.convert.ConverterImpl;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.metadata.RelMetadataQuery;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexInputRef;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexProgram;
import org.apache.calcite.schema.*;
import org.apache.calcite.sql.SqlCall;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.util.BuiltInMethod;

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataQuery;
import ch.epfl.dias.repl.Repl;
import org.json4s.JsonAST;

import com.google.common.base.Function;
import com.google.common.collect.Lists;
import org.apache.calcite.util.Sources;

import java.io.File;
import java.util.List;

/**
 * Relational expression representing a scan of a table in a Cassandra data source.
 */
public class PelagoToEnumerableConverter
        extends ConverterImpl
        implements EnumerableRel { //ScannableTable
//    CsvTranslatableTable csvTable;

    private PelagoToEnumerableConverter(
            RelOptCluster cluster,
            RelTraitSet traits,
            RelNode input) {
        super(cluster, ConventionTraitDef.INSTANCE, traits, input);
//        csvTable = new CsvTranslatableTable(Sources.of(new File("/home/periklis/Desktop/test.csv")), RelDataTypeImpl.proto(getRowType()));
    }

    @Override public RelNode copy(RelTraitSet traitSet, List<RelNode> inputs) {
        return copy(traitSet, sole(inputs));
    }

    public RelNode copy(RelTraitSet traitSet, RelNode input) {
        return PelagoToEnumerableConverter.create(input);
    }

    public static RelNode create(RelNode input){
        RelOptCluster cluster  = input.getCluster();
        RelTraitSet traitSet = input.getTraitSet().replace(EnumerableConvention.INSTANCE)
            .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                public RelDistribution get() {
                    return cluster.getMetadataQuery().distribution(input);
                }
            })
            .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                public RelDeviceType get() {
                    return ((PelagoRelMetadataQuery) cluster.getMetadataQuery()).deviceType(input);
                }
            });
        return new PelagoToEnumerableConverter(input.getCluster(), traitSet, input);
    }

    @Override public RelOptCost computeSelfCost(RelOptPlanner planner,
                                                RelMetadataQuery mq) {
        return super.computeSelfCost(planner, mq)
                .multiplyBy(((double) getRowType().getFieldCount()) * 0.1);


//        return super.computeSelfCost(planner, mq).multiplyBy(.1);
    }

//    private static RelProtoDataType rproto;
    private static PelagoResultTable pt;

    @SuppressWarnings("UnusedDeclaration")
    public static Enumerable getEnumerableResult(DataContext root){
        return pt.scan(root);
    }

    public Result implement(EnumerableRelImplementor implementor, Prefer pref) {
        boolean mock = true;    //TODO: change!!!

        pt = new PelagoResultTable(Sources.of(new File(Repl.mockfile())), getRowType(), mock); //TODO: fix path

        RelOptTable table = RelOptTableImpl.create(null, getRowType(), ImmutableList.of(),
                Expressions.call(
                        Types.lookupMethod(PelagoToEnumerableConverter.class, "getEnumerableResult", DataContext.class),
                        DataContext.ROOT
                )
        );

        final int[] fields = new int[getRowType().getFieldCount()];
        for (int i = 0; i < fields.length; i++) fields[i] = i;

        JsonAST.JValue plan = ((PelagoRel) getInput()).implement()._2;
//        System.out.println(JsonMethods$.MODULE$.pretty(JsonMethods$.MODULE$.render(plan, PlanToJSON.formats())));

        PelagoResultScan ts = new PelagoResultScan(getCluster(), table, pt, fields);
        return ts.implement(implementor, pref);
    }

    /** E.g. {@code constantArrayList("x", "y")} returns
     * "Arrays.asList('x', 'y')". */
    private static <T> MethodCallExpression constantArrayList(List<T> values,
                                                              Class clazz) {
        return Expressions.call(
                BuiltInMethod.ARRAYS_AS_LIST.method,
                Expressions.newArrayInit(clazz, constantList(values)));
    }

    /** E.g. {@code constantList("x", "y")} returns
     * {@code {ConstantExpression("x"), ConstantExpression("y")}}. */
    private static <T> List<Expression> constantList(List<T> values) {
        return Lists.transform(values,
                new Function<T, Expression>() {
                    public Expression apply(T a0) {
                        return Expressions.constant(a0);
                    }
                });
    }

    public RelWriter explainTerms(RelWriter pw){
        return super.explainTerms(pw).item("trait", getTraitSet().toString());
    }


//    public Enumerable<Object[]> scan(DataContext root) {
//        final int[] fields = CsvEnumerator.identityList(fieldTypes.size());
//        final AtomicBoolean cancelFlag = DataContext.Variable.CANCEL_FLAG.get(root);
//        return new AbstractEnumerable<Object[]>() {
//            public Enumerator<Object[]> enumerator() {
//                return new CsvEnumerator<>(source, cancelFlag, false, null,
//                        new CsvEnumerator.ArrayRowConverter(fieldTypes, fields));
//            }
//        };
//    }
//
//    @Override
//    public RelDataType getRowType(RelDataTypeFactory typeFactory) {
//        return this.getRowType();
//    }
//
//    @Override
//    public Statistic getStatistic() {
//        return Statistics.UNKNOWN;
//    }
//
//    @Override
//    public Schema.TableType getJdbcTableType() {
//        return Schema.TableType.TABLE;
//    }
//
//    @Override
//    public boolean isRolledUp(String column) {
//        return false;
//    }
//
//    @Override
//    public boolean rolledUpColumnValidInsideAgg(String column, SqlCall call, SqlNode parent, CalciteConnectionConfig config) {
//        return true;
//    }
//
//    public Expression getExpression(SchemaPlus schema, String tableName,
//                                    Class clazz) {
//        return Schemas.tableExpression(schema, getElementType(), tableName, clazz);
//    }
//
//    public Type getElementType() {
//        return Object[].class;
//    }
//
//    public <T> Queryable<T> asQueryable(QueryProvider queryProvider,
//                                        SchemaPlus schema, String tableName) {
//        throw new UnsupportedOperationException();
//    }
//
//    public RelNode toRel(
//            RelOptTable.ToRelContext context,
//            RelOptTable relOptTable) {
//        // Request all fields.
//        final int fieldCount = relOptTable.getRowType().getFieldCount();
//        final int[] fields = CsvEnumerator.identityList(fieldCount);
//        return new CsvTableScan(context.getCluster(), relOptTable, csvTable, fields);
//    }
}
