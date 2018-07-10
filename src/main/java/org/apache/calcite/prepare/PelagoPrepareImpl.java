package org.apache.calcite.prepare;

//import ch.epfl.dias.calcite.adapter.pelago.trait.RelDeviceTypeTraitDef;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import org.apache.calcite.DataContext;
import org.apache.calcite.adapter.enumerable.EnumerableConvention;
import org.apache.calcite.adapter.enumerable.EnumerableRel;
import org.apache.calcite.adapter.enumerable.EnumerableRules;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.avatica.AvaticaParameter;
import org.apache.calcite.avatica.ColumnMetaData;
import org.apache.calcite.avatica.Meta;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.interpreter.BindableConvention;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.linq4j.Enumerable;
import org.apache.calcite.linq4j.Linq4j;
import org.apache.calcite.linq4j.Ord;
import org.apache.calcite.linq4j.function.Function1;
import org.apache.calcite.plan.*;
import org.apache.calcite.plan.volcano.AbstractConverter;
import org.apache.calcite.plan.volcano.VolcanoPlanner;
import org.apache.calcite.prepare.PelagoPreparingStmt;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelDistributionTraitDef;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.Calc;
import org.apache.calcite.rel.core.Filter;
import org.apache.calcite.rel.core.Join;
import org.apache.calcite.rel.core.JoinRelType;
import org.apache.calcite.rel.core.Project;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.core.Sort;
import org.apache.calcite.rel.core.TableScan;
import org.apache.calcite.rel.core.Values;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalFilter;
import org.apache.calcite.rel.logical.LogicalJoin;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.logical.LogicalSort;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.runtime.Bindable;
import org.apache.calcite.runtime.Hook;
import org.apache.calcite.runtime.Typed;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.SqlOperatorTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserImplFactory;
import org.apache.calcite.sql.type.ExtraSqlTypes;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.ChainedSqlOperatorTable;
import org.apache.calcite.sql.validate.SqlConformance;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.tools.Program;
import org.apache.calcite.tools.Programs;
import org.apache.calcite.util.Holder;
import org.apache.calcite.util.Util;

import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceTypeTraitDef;
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataProvider;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoRules;

import java.lang.reflect.Type;
import java.sql.DatabaseMetaData;
import java.sql.Types;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.apache.calcite.plan.RelOptRule.any;
import static org.apache.calcite.plan.RelOptRule.none;
import static org.apache.calcite.plan.RelOptRule.operand;
import static org.apache.calcite.plan.RelOptRule.some;

public class PelagoPrepareImpl extends CalcitePrepareImpl {
    /** Creates a query planner and initializes it with a default set of
     * rules. */
    protected RelOptPlanner createPlanner(CalcitePrepare.Context prepareContext) {
        return createPlanner(prepareContext, null, null);
    }

    /** Creates a query planner and initializes it with a default set of
     * rules. */
    protected RelOptPlanner createPlanner(
            final CalcitePrepare.Context prepareContext,
            org.apache.calcite.plan.Context externalContext,
            RelOptCostFactory costFactory) {
        RelOptPlanner planner = super.createPlanner(prepareContext, externalContext, costFactory);
        planner.addRelTraitDef(RelDistributionTraitDef.INSTANCE);
        planner.addRelTraitDef(RelDeviceTypeTraitDef  .INSTANCE);

//        COMMUTE
//                ? JoinAssociateRule.INSTANCE
//                : ProjectMergeRule.INSTANCE,
//        planner.removeRule(JoinAssociateRule.INSTANCE);
        for (RelOptRule r: planner.getRules()){
            planner.removeRule(r);
        }
        //FIXME: not so certain any more about which RelFactory we should use and which rules should be applied to the core RelNodes oand which ones to the Logical ones.\
        //this may as well be disabling some rules...
        ((VolcanoPlanner) planner).registerAbstractRelationalRules();
        planner.addRule(AbstractConverter.ExpandConversionRule.INSTANCE);
//        System.out.println(planner.getRules());

        for (RelOptRule rule: PelagoRules.RULES) {
            planner.addRule(rule);
        }

        List<RelOptRule> rules = new ArrayList<RelOptRule>();
        rules.add(new TableScanRule(PelagoRelFactories.PELAGO_BUILDER));
        // push and merge filter rules
        rules.add(new FilterAggregateTransposeRule(Filter.class, PelagoRelFactories.PELAGO_BUILDER, Aggregate.class));
//        rules.add(new FilterProjectTransposeRule  (Filter.class, Project.class, true, true, PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new FilterMergeRule(PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new FilterJoinRule.FilterIntoJoinRule(true, PelagoRelFactories.PELAGO_BUILDER,
            new FilterJoinRule.Predicate() {
                public boolean apply(Join join, JoinRelType joinType, RexNode exp) {
                    return true;
                }
            }));
        rules.add(new FilterJoinRule.JoinConditionPushRule(PelagoRelFactories.PELAGO_BUILDER,  new FilterJoinRule.Predicate() {
            public boolean apply(Join join, JoinRelType joinType, RexNode exp) {
                return true;
            }
        }));
        /*push filter into the children of a join*/
        rules.add(FilterTableScanRule.INSTANCE);
        // push and merge projection rules
        rules.add(new ProjectRemoveRule(PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new ProjectJoinTransposeRule(PushProjector.ExprCondition.TRUE, PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new JoinProjectTransposeRule(
            RelOptRule.operand(Join.class,
                RelOptRule.operand(Project.class, RelOptRule.any()),
                RelOptRule.operand(Project.class, RelOptRule.any())),
            "JoinProjectTransposeRule(Project-Project)",
            false, PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new ProjectFilterTransposeRule(
            LogicalProject.class, LogicalFilter.class, PelagoRelFactories.PELAGO_BUILDER,
            PushProjector.ExprCondition.FALSE)); //XXX causes non-termination
        /*it is better to use filter first an then project*/
        rules.add(ProjectTableScanRule.INSTANCE);
        rules.add(new ProjectMergeRule(true, PelagoRelFactories.PELAGO_BUILDER));
        //aggregate rules
        rules.add(AggregateRemoveRule.INSTANCE);
        rules.add(AggregateJoinTransposeRule.INSTANCE);
        rules.add(new AggregateProjectMergeRule(Aggregate.class, Project.class, PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new AggregateProjectPullUpConstantsRule(Aggregate.class,
            Project.class, PelagoRelFactories.PELAGO_BUILDER,
            "AggregateProjectPullUpConstantsRule"));
        rules.add(new AggregateExpandDistinctAggregatesRule(Aggregate.class, true, PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new AggregateReduceFunctionsRule(RelOptRule.operand(Aggregate.class, any()), PelagoRelFactories.PELAGO_BUILDER)); //optimizes out required sorting in some cases!
        //join rules
//                                                                                                                rules.add(JoinToMultiJoinRule.INSTANCE);
//                                                                                                                rules.add(LoptOptimizeJoinRule.INSTANCE);
//                                                                                                              rules.add(MultiJoinOptimizeBushyRule.INSTANCE);//,
                                                                                                                rules.add(JoinPushThroughJoinRule.LEFT );
                                                                                                                rules.add(JoinPushThroughJoinRule.RIGHT);
                                                                                                                /*choose between right and left*/
                                                                                                                rules.add(JoinPushExpressionsRule.INSTANCE);
                                                                                                                rules.add(JoinAssociateRule.INSTANCE);
                                                                                                                rules.add(JoinCommuteRule.INSTANCE);
        // simplify expressions rules
        rules.add(new ReduceExpressionsRule.CalcReduceExpressionsRule(Calc.class, true,
            PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new ReduceExpressionsRule.FilterReduceExpressionsRule(Filter.class, true,
            PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new ReduceExpressionsRule.ProjectReduceExpressionsRule(Project.class, true,
            PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new ReduceExpressionsRule.JoinReduceExpressionsRule(Join.class, true,
            PelagoRelFactories.PELAGO_BUILDER));
        // prune empty results rules
        rules.add(new PruneEmptyRules.RemoveEmptySingleRule(Filter.class, "PruneEmptyFilter"));
        rules.add(new PruneEmptyRules.RemoveEmptySingleRule(Project.class, Predicates.<Project>alwaysTrue(),
            PelagoRelFactories.PELAGO_BUILDER, "PruneEmptyProject"));
        rules.add(new PruneEmptyRules.RemoveEmptySingleRule(Aggregate.class, Aggregate.IS_NOT_GRAND_TOTAL,
            PelagoRelFactories.PELAGO_BUILDER, "PruneEmptyAggregate"));
        rules.add(PruneEmptyRules.JOIN_LEFT_INSTANCE);
        rules.add(PruneEmptyRules.JOIN_RIGHT_INSTANCE);
        rules.add(ProjectTableScanRule.INSTANCE);
        rules.add(new AggregateProjectPullUpConstantsRule(Aggregate.class,
            RelNode.class, PelagoRelFactories.PELAGO_BUILDER,
            "AggregatePullUpConstantsRule"));
        /* Sort Rules*/
        rules.add(new SortJoinTransposeRule(Sort.class,
            Join.class, PelagoRelFactories.PELAGO_BUILDER));
        rules.add(new SortProjectTransposeRule(Sort.class, Project.class,
            PelagoRelFactories.PELAGO_BUILDER, null));
        //SortRemoveRule.INSTANCE, //Too aggressive when triggered over enumerables; always removes Sort
        rules.add(SortUnionTransposeRule.INSTANCE);
        /*Enumerable Rules*/
//        rules.add(EnumerableRules.ENUMERABLE_FILTER_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_TABLE_SCAN_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_PROJECT_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_AGGREGATE_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_JOIN_RULE);
//      rules.add(EnumerableRules.ENUMERABLE_MERGE_JOIN_RULE) //FIMXE: no mergejoin yet
        rules.add(EnumerableRules.ENUMERABLE_SEMI_JOIN_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_SORT_RULE);       //FIMXE: no support for SORT yet
//      rules.add(EnumerableRules.ENUMERABLE_UNION_RULE)      //FIMXE: no support for UNION yet
//      rules.add(EnumerableRules.ENUMERABLE_INTERSECT_RULE)  //FIMXE: no support for INTERSECT yet
//      rules.add(EnumerableRules.ENUMERABLE_MINUS_RULE)      //FIMXE: no support for MINUS yet
        rules.add(EnumerableRules.ENUMERABLE_COLLECT_RULE);
        rules.add(EnumerableRules.ENUMERABLE_UNCOLLECT_RULE);
        rules.add(EnumerableRules.ENUMERABLE_CORRELATE_RULE);
        rules.add(EnumerableRules.ENUMERABLE_VALUES_RULE);
//      rules.add(EnumerableRules.ENUMERABLE_JOIN_RULE)
//
//      rules.add(EnumerableRules.ENUMERABLE_MERGE_JOIN_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_SEMI_JOIN_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_CORRELATE_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_PROJECT_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_FILTER_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_AGGREGATE_RULE);
//      rules.add(EnumerableRules.ENUMERABLE_SORT_RULE);
//      rules.add(EnumerableRules.ENUMERABLE_LIMIT_RULE)
        rules.add(EnumerableRules.ENUMERABLE_UNION_RULE);
        rules.add(EnumerableRules.ENUMERABLE_INTERSECT_RULE);
        rules.add(EnumerableRules.ENUMERABLE_MINUS_RULE);
        rules.add(EnumerableRules.ENUMERABLE_TABLE_MODIFICATION_RULE);
//      rules.add(EnumerableRules.ENUMERABLE_VALUES_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_WINDOW_RULE)

        for (RelOptRule r: rules){
            planner.addRule(r);
        }






//        planner.removeRule(ProjectMergeRule.INSTANCE);
//        planner.addRule(JoinAssociateRule.INSTANCE);
//        planner.addRule(JoinToMultiJoinRule.INSTANCE);
//        planner.addRule(LoptOptimizeJoinRule.INSTANCE);
//        //        MultiJoinOptimizeBushyRule.INSTANCE,
//        planner.addRule(JoinPushThroughJoinRule.RIGHT);
//        planner.addRule(JoinPushThroughJoinRule.LEFT);
//        /*choose between right and left*/
//        planner.addRule(JoinPushExpressionsRule.INSTANCE);
//        planner.removeRule(EnumerableRules.ENUMERABLE_PROJECT_TO_CALC_RULE);
//        System.out.println(planner.getRules());
        return planner;
    }

    /** Creates a collection of planner factories.
     *
     * <p>The collection must have at least one factory, and each factory must
     * create a planner. If the collection has more than one planner, Calcite will
     * try each planner in turn.</p>
     *
     * <p>One of the things you can do with this mechanism is to try a simpler,
     * faster, planner with a smaller rule set first, then fall back to a more
     * complex planner for complex and costly queries.</p>
     *
     * <p>The default implementation returns a factory that calls
     * {@link #createPlanner(org.apache.calcite.jdbc.CalcitePrepare.Context)}.</p>
     */
    protected List<Function1<Context, RelOptPlanner>> createPlannerFactories() {
        return Collections.<Function1<Context, RelOptPlanner>>singletonList(
                new Function1<Context, RelOptPlanner>() {
                    public RelOptPlanner apply(Context context) {
                        return createPlanner(context, null, null);
                    }
                });
    }


    /** Factory method for cluster. */
    protected RelOptCluster createCluster(RelOptPlanner planner,
        RexBuilder rexBuilder) {

        RelOptCluster cluster = PelagoRelOptCluster.create(planner, rexBuilder);//super.createCluster(planner, rexBuilder);
        cluster.setMetadataProvider(PelagoRelMetadataProvider.INSTANCE);
        return cluster;
    }



    /**
     * TODO: Remove the following duplicate code by changing the invoked programs "the correct way"
     * Everything below this point is copied from Calcite just to manage to change the invoked Programs.
     * There is for sure some better way to do it...
     *Find it and remove the following code together with {@link PelagoPreparingSmt}
     */






    /**
     * Deduces the broad type of statement.
     * Currently returns SELECT for most statement types, but this may change.
     *
     * @param kind Kind of statement
     */
    private Meta.StatementType getStatementType(SqlKind kind) {
        switch (kind) {
            case INSERT:
            case DELETE:
            case UPDATE:
                return Meta.StatementType.IS_DML;
            case CREATE_TABLE:
                return Meta.StatementType.CREATE;
            default:
                return Meta.StatementType.SELECT;
        }
    }

    private SqlValidator createSqlValidator(Context context,
                                            CalciteCatalogReader catalogReader) {
        final SqlOperatorTable opTab0 =
                context.config().fun(SqlOperatorTable.class,
                        SqlStdOperatorTable.instance());
        final SqlOperatorTable opTab =
                ChainedSqlOperatorTable.of(opTab0, catalogReader);
        final JavaTypeFactory typeFactory = context.getTypeFactory();
        final SqlConformance conformance = context.config().conformance();
        return new CalciteSqlValidator(opTab, catalogReader, typeFactory,
                conformance);
    }

    /**
     * Deduces the broad type of statement for a prepare result.
     * Currently returns SELECT for most statement types, but this may change.
     *
     * @param preparedResult Prepare result
     */
    private Meta.StatementType getStatementType(Prepare.PreparedResult preparedResult) {
        if (preparedResult.isDml()) {
            return Meta.StatementType.IS_DML;
        } else {
            return Meta.StatementType.SELECT;
        }
    }

    private int getTypeOrdinal(RelDataType type) {
        return type.getSqlTypeName().getJdbcOrdinal();
    }

    private static String getClassName(RelDataType type) {
        return null;
    }

    private static int getScale(RelDataType type) {
        return type.getScale() == RelDataType.SCALE_NOT_SPECIFIED
                ? 0
                : type.getScale();
    }

    private static int getPrecision(RelDataType type) {
        return type.getPrecision() == RelDataType.PRECISION_NOT_SPECIFIED
                ? 0
                : type.getPrecision();
    }

    /** Returns the type name in string form. Does not include precision, scale
     * or whether nulls are allowed. Example: "DECIMAL" not "DECIMAL(7, 2)";
     * "INTEGER" not "JavaType(int)". */
    private static String getTypeName(RelDataType type) {
        final SqlTypeName sqlTypeName = type.getSqlTypeName();
        switch (sqlTypeName) {
            case ARRAY:
            case MULTISET:
            case MAP:
            case ROW:
                return type.toString(); // e.g. "INTEGER ARRAY"
            case INTERVAL_YEAR_MONTH:
                return "INTERVAL_YEAR_TO_MONTH";
            case INTERVAL_DAY_HOUR:
                return "INTERVAL_DAY_TO_HOUR";
            case INTERVAL_DAY_MINUTE:
                return "INTERVAL_DAY_TO_MINUTE";
            case INTERVAL_DAY_SECOND:
                return "INTERVAL_DAY_TO_SECOND";
            case INTERVAL_HOUR_MINUTE:
                return "INTERVAL_HOUR_TO_MINUTE";
            case INTERVAL_HOUR_SECOND:
                return "INTERVAL_HOUR_TO_SECOND";
            case INTERVAL_MINUTE_SECOND:
                return "INTERVAL_MINUTE_TO_SECOND";
            default:
                return sqlTypeName.getName(); // e.g. "DECIMAL", "INTERVAL_YEAR_MONTH"
        }
    }

    private static RelDataType makeStruct(
            RelDataTypeFactory typeFactory,
            RelDataType type) {
        if (type.isStruct()) {
            return type;
        }
        return typeFactory.builder().add("$0", type).build();
    }

    private static String origin(List<String> origins, int offsetFromEnd) {
        return origins == null || offsetFromEnd >= origins.size()
                ? null
                : origins.get(origins.size() - 1 - offsetFromEnd);
    }

    private ColumnMetaData.AvaticaType avaticaType(JavaTypeFactory typeFactory,
                                                   RelDataType type, RelDataType fieldType) {
        final String typeName = getTypeName(type);
        if (type.getComponentType() != null) {
            final ColumnMetaData.AvaticaType componentType =
                    avaticaType(typeFactory, type.getComponentType(), null);
            final Type clazz = typeFactory.getJavaClass(type.getComponentType());
            final ColumnMetaData.Rep rep = ColumnMetaData.Rep.of(clazz);
            assert rep != null;
            return ColumnMetaData.array(componentType, typeName, rep);
        } else {
            int typeOrdinal = getTypeOrdinal(type);
            switch (typeOrdinal) {
                case Types.STRUCT:
                    final List<ColumnMetaData> columns = new ArrayList<>();
                    for (RelDataTypeField field : type.getFieldList()) {
                        columns.add(
                                metaData(typeFactory, field.getIndex(), field.getName(),
                                        field.getType(), null, null));
                    }
                    return ColumnMetaData.struct(columns);
                case ExtraSqlTypes.GEOMETRY:
                    typeOrdinal = Types.VARCHAR;
                    // fall through
                default:
                    final Type clazz =
                            typeFactory.getJavaClass(Util.first(fieldType, type));
                    final ColumnMetaData.Rep rep = ColumnMetaData.Rep.of(clazz);
                    assert rep != null;
                    return ColumnMetaData.scalar(typeOrdinal, typeName, rep);
            }
        }
    }

    private ColumnMetaData metaData(JavaTypeFactory typeFactory, int ordinal,
                                    String fieldName, RelDataType type, RelDataType fieldType,
                                    List<String> origins) {
        final ColumnMetaData.AvaticaType avaticaType =
                avaticaType(typeFactory, type, fieldType);
        return new ColumnMetaData(
                ordinal,
                false,
                true,
                false,
                false,
                type.isNullable()
                        ? DatabaseMetaData.columnNullable
                        : DatabaseMetaData.columnNoNulls,
                true,
                type.getPrecision(),
                fieldName,
                origin(origins, 0),
                origin(origins, 2),
                getPrecision(type),
                getScale(type),
                origin(origins, 1),
                null,
                avaticaType,
                true,
                false,
                false,
                avaticaType.columnClassName());
    }

    private List<ColumnMetaData> getColumnMetaDataList(
            JavaTypeFactory typeFactory, RelDataType x, RelDataType jdbcType,
            List<List<String>> originList) {
        final List<ColumnMetaData> columns = new ArrayList<>();
        for (Ord<RelDataTypeField> pair : Ord.zip(jdbcType.getFieldList())) {
            final RelDataTypeField field = pair.e;
            final RelDataType type = field.getType();
            final RelDataType fieldType =
                    x.isStruct() ? x.getFieldList().get(pair.i).getType() : type;
            columns.add(
                    metaData(typeFactory, columns.size(), field.getName(), type,
                            fieldType, originList.get(pair.i)));
        }
        return columns;
    }

    @Override
    <T> CalciteSignature<T> prepare2_(
            Context context,
            Query<T> query,
            Type elementType,
            long maxRowCount,
            CalciteCatalogReader catalogReader,
            RelOptPlanner planner) {
        final JavaTypeFactory typeFactory = context.getTypeFactory();
        final EnumerableRel.Prefer prefer;
        if (elementType == Object[].class) {
            prefer = EnumerableRel.Prefer.ARRAY;
        } else {
            prefer = EnumerableRel.Prefer.CUSTOM;
        }
        final Convention resultConvention =
                enableBindable ? BindableConvention.INSTANCE
                        : EnumerableConvention.INSTANCE;
        final PelagoPreparingStmt preparingStmt =
                new PelagoPreparingStmt(this, context, catalogReader, typeFactory,
                        context.getRootSchema(), prefer, planner, resultConvention,
                        createConvertletTable());

        final RelDataType x;
        final Prepare.PreparedResult preparedResult;
        final Meta.StatementType statementType;
        if (query.sql != null) {
            final CalciteConnectionConfig config = context.config();
            final SqlParser.ConfigBuilder parserConfig = createParserConfig()
                    .setQuotedCasing(config.quotedCasing())
                    .setUnquotedCasing(config.unquotedCasing())
                    .setQuoting(config.quoting())
                    .setConformance(config.conformance())
                    .setCaseSensitive(config.caseSensitive());
            final SqlParserImplFactory parserFactory =
                    config.parserFactory(SqlParserImplFactory.class, null);
            if (parserFactory != null) {
                parserConfig.setParserFactory(parserFactory);
            }
            SqlParser parser = createParser(query.sql,  parserConfig);
            SqlNode sqlNode;
            try {
                sqlNode = parser.parseStmt();
                statementType = getStatementType(sqlNode.getKind());
            } catch (SqlParseException e) {
                throw new RuntimeException(
                        "parse failed: " + e.getMessage(), e);
            }

            Hook.PARSE_TREE.run(new Object[] {query.sql, sqlNode});

            if (sqlNode.getKind().belongsTo(SqlKind.DDL)) {
                executeDdl(context, sqlNode);

                return new CalciteSignature<>(query.sql,
                        ImmutableList.<AvaticaParameter>of(),
                        ImmutableMap.<String, Object>of(), null,
                        ImmutableList.<ColumnMetaData>of(), Meta.CursorFactory.OBJECT,
                        null, ImmutableList.<RelCollation>of(), -1, null,
                        Meta.StatementType.OTHER_DDL);
            }

            final SqlValidator validator =
                    createSqlValidator(context, catalogReader);
            validator.setIdentifierExpansion(true);
            validator.setDefaultNullCollation(config.defaultNullCollation());

            preparedResult = preparingStmt.prepareSql(
                    sqlNode, Object.class, validator, true);
            switch (sqlNode.getKind()) {
                case INSERT:
                case DELETE:
                case UPDATE:
                case EXPLAIN:
                    // FIXME: getValidatedNodeType is wrong for DML
                    x = RelOptUtil.createDmlRowType(sqlNode.getKind(), typeFactory);
                    break;
                default:
                    x = validator.getValidatedNodeType(sqlNode);
            }
        } else if (query.queryable != null) {
            x = context.getTypeFactory().createType(elementType);
            preparedResult =
                    preparingStmt.prepareQueryable(query.queryable, x);
            statementType = getStatementType(preparedResult);
        } else {
            assert query.rel != null;
            x = query.rel.getRowType();
            preparedResult = preparingStmt.prepareRel(query.rel);
            statementType = getStatementType(preparedResult);
        }

        final List<AvaticaParameter> parameters = new ArrayList<>();
        final RelDataType parameterRowType = preparedResult.getParameterRowType();
        for (RelDataTypeField field : parameterRowType.getFieldList()) {
            RelDataType type = field.getType();
            parameters.add(
                    new AvaticaParameter(
                            false,
                            getPrecision(type),
                            getScale(type),
                            getTypeOrdinal(type),
                            getTypeName(type),
                            getClassName(type),
                            field.getName()));
        }

        RelDataType jdbcType = makeStruct(typeFactory, x);
        final List<List<String>> originList = preparedResult.getFieldOrigins();
        final List<ColumnMetaData> columns =
                getColumnMetaDataList(typeFactory, x, jdbcType, originList);
        Class resultClazz = null;
        if (preparedResult instanceof Typed) {
            resultClazz = (Class) ((Typed) preparedResult).getElementType();
        }
        final Meta.CursorFactory cursorFactory =
                preparingStmt.resultConvention == BindableConvention.INSTANCE
                        ? Meta.CursorFactory.ARRAY
                        : Meta.CursorFactory.deduce(columns, resultClazz);
        //noinspection unchecked
        final Bindable<T> bindable = preparedResult.getBindable(cursorFactory);
        return new CalciteSignature<>(
                query.sql,
                parameters,
                preparingStmt.getInternalParameters(),
                jdbcType,
                columns,
                cursorFactory,
                context.getRootSchema(),
                preparedResult instanceof Prepare.PreparedResultImpl
                        ? ((Prepare.PreparedResultImpl) preparedResult).collations
                        : ImmutableList.<RelCollation>of(),
                maxRowCount,
                bindable,
                statementType);
    }
}
