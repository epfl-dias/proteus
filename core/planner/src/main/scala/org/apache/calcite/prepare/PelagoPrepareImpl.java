package org.apache.calcite.prepare;

import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataProvider;
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoTableScan;
import ch.epfl.dias.calcite.adapter.pelago.rel.PelagoUnpack;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoProjectPushBelowUnpack;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoRules;
import ch.epfl.dias.calcite.adapter.pelago.traits.*;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import org.apache.calcite.adapter.enumerable.*;
import org.apache.calcite.adapter.java.JavaTypeFactory;
import org.apache.calcite.avatica.AvaticaParameter;
import org.apache.calcite.avatica.ColumnMetaData;
import org.apache.calcite.avatica.Meta;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.config.CalciteSystemProperty;
import org.apache.calcite.interpreter.BindableConvention;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.linq4j.Linq4j;
import org.apache.calcite.linq4j.Ord;
import org.apache.calcite.linq4j.function.Function1;
import org.apache.calcite.plan.*;
import org.apache.calcite.plan.volcano.AbstractConverter;
import org.apache.calcite.plan.volcano.PelagoCostFactory;
import org.apache.calcite.plan.volcano.VolcanoPlanner;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelCollationTraitDef;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.RelDataTypeField;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.runtime.Bindable;
import org.apache.calcite.runtime.Hook;
import org.apache.calcite.runtime.Typed;
import org.apache.calcite.sql.*;
import org.apache.calcite.sql.ddl.SqlCreatePelagoTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParseException;
import org.apache.calcite.sql.parser.SqlParser;
import org.apache.calcite.sql.parser.SqlParserImplFactory;
import org.apache.calcite.sql.pretty.SqlPrettyWriter;
import org.apache.calcite.sql.type.ExtraSqlTypes;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.util.SqlOperatorTables;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.tools.*;
import org.apache.calcite.util.Pair;
import org.apache.calcite.util.Util;

import ch.epfl.dias.repl.Repl;

import java.io.PrintStream;
import java.lang.reflect.Type;
import java.sql.DatabaseMetaData;
import java.sql.Types;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.apache.calcite.plan.RelOptRule.any;
import static org.apache.calcite.plan.RelOptRule.operand;

public class PelagoPrepareImpl extends CalcitePrepareImpl {

    public PelagoPrepareImpl() {
        super();


    }

    /** Creates a query planner and initializes it with a default set of
     * rules. */
    public RelOptPlanner createPlanner(CalcitePrepare.Context prepareContext) {
        return createPlanner(prepareContext, null, null);
    }

    /** Creates a query planner and initializes it with a default set of
     * rules. */
    protected RelOptPlanner createPlanner(
            final CalcitePrepare.Context prepareContext,
            org.apache.calcite.plan.Context externalContext,
            RelOptCostFactory costFactory) {
        RelOptCostFactory cFactory = (costFactory == null) ? PelagoCostFactory.INSTANCE : costFactory;

        if (externalContext == null) {
            externalContext = Contexts.of(prepareContext.config());
        }
        final VolcanoPlanner planner =
            new VolcanoPlanner(cFactory, externalContext){

//                public RelOptCost getCost(RelNode rel, RelMetadataQuery mq) {
////                    System.out.println(rel);
//                    if (rel instanceof RelSubset) {
//                        return ((RelSubset) rel).computeSelfCost(this, mq);
//                    }
//                    return super.getCost(rel, mq);
//                }
            };
        planner.addRelTraitDef(ConventionTraitDef.INSTANCE);
        if (false) planner.addListener(new RelOptListener() {
            final HashMap<RelOptRule, Long> invocations = new HashMap<>();
            final HashMap<RelOptRule, Long> transforms = new HashMap<>();
            final HashMap<Pair<RelTraitSet, RelTraitSet>, Long> expands = new HashMap<>();
            final HashMap<RelTraitSet, Long> expandsFrom = new HashMap<>();
            final HashMap<RelTraitSet, Long> expandsTo = new HashMap<>();

            @Override public void relEquivalenceFound(final RelEquivalenceEvent event) {

            }

            public <T> void increment(HashMap<T, Long> map, T key){
                map.put(key, map.getOrDefault(key, 0L) + 1);
            }

            @Override public void ruleAttempted(final RuleAttemptedEvent event) {
                if (event.isBefore()) return;
                increment(invocations, event.getRuleCall().rule);
                if (event.getRuleCall().getRule() == AbstractConverter.ExpandConversionRule.INSTANCE){
                    increment(expands, Pair.of(event.getRel().getInput(0).getTraitSet(), event.getRel().getTraitSet()));
                    increment(expandsFrom, event.getRel().getInput(0).getTraitSet());
                    increment(expandsTo, event.getRel().getTraitSet());
                }
            }

            @Override public void ruleProductionSucceeded(final RuleProductionEvent event) {
                if (event.isBefore()) return;
                increment(transforms, event.getRuleCall().rule);
            }

            @Override public void relDiscarded(final RelDiscardedEvent event) {

            }

            public String repeat(String x, int n){
                return IntStream.range(0, n).mapToObj(i -> x).collect(Collectors.joining(""));
            }

            public <T> void printBarChart(PrintStream out, HashMap<T, Long> maps, int barWidth){
                long counter = maps.values().stream().mapToLong(Long::longValue).sum();
                int numSize = Math.max((int) Math.ceil(Math.log10(counter)), 1);
                maps.entrySet().stream().sorted(Comparator.comparing(Object::toString)).forEach(x -> {
                    int barSize = (int) (barWidth * 1.0 * x.getValue() / counter);
                    out.print(repeat("#", barSize));
                    out.print(repeat(" ", barWidth - barSize));
                    out.format("\t%" + numSize + "d/%d\t", x.getValue(), counter);
                    out.print(x.getKey());
                    out.println();
                });
                System.out.println("Total: " + counter);
            }

            @Override public void relChosen(final RelChosenEvent event) {
                if (event.getRel() == null) {
                    int barWidth = 30;

                    printBarChart(System.out, invocations, barWidth);
                    printBarChart(System.out, expands, barWidth);
                    printBarChart(System.out, transforms, barWidth);

                    printBarChart(System.out, expandsFrom, barWidth);
                    printBarChart(System.out, expandsTo, barWidth);

                    transforms.clear();
                    invocations.clear();
                    expands.clear();
                }
            }
        });
        if (CalciteSystemProperty.ENABLE_COLLATION_TRAIT.value()) {
            planner.addRelTraitDef(RelCollationTraitDef.INSTANCE);
        }
        RelOptUtil.registerDefaultRules(planner,
            prepareContext.config().materializationsEnabled(),
            enableBindable);
//        Hook.PLANNER.run(planner); // allow test to add or remove rules

//        RelOptPlanner planner = super.createPlanner(prepareContext, externalContext, cFactory);
        planner.addRelTraitDef(RelPackingTraitDef.INSTANCE());
        planner.addRelTraitDef(RelDeviceTypeTraitDef     .INSTANCE());
        planner.addRelTraitDef(RelHomDistributionTraitDef.INSTANCE());

        planner.addRelTraitDef(RelHetDistributionTraitDef.INSTANCE());
        if (Repl.isHybrid()) planner.addRelTraitDef(RelSplitPointTraitDef.INSTANCE());
        if (Repl.isHybrid() || Repl.isCpuonly()) planner.addRelTraitDef(RelComputeDeviceTraitDef  .INSTANCE());
//
//        planner.clear();
//
//        //FIXME: not so certain any more about which RelFactory we should use and which rules should be applied to the core RelNodes oand which ones to the Logical ones.\
//        //this may as well be disabling some rules...
//        ((VolcanoPlanner) planner).registerAbstractRelationalRules();
//
//        planner.addRule(AbstractConverter.ExpandConversionRule.INSTANCE);
////        System.out.println(planner.getRules());
        for (RelOptRule rule: PelagoRules.RULES()) {
            planner.addRule(rule);
        }
        for (var enrule: EnumerableRules.ENUMERABLE_RULES){
            planner.removeRule(enrule);
        }
        planner.removeRule(CoreRules.JOIN_COMMUTE);

        planner.removeRule(EnumerableRules.TO_INTERPRETER);
        planner.addRule(EnumerableRules.ENUMERABLE_PROJECT_TO_CALC_RULE);
        planner.addRule(EnumerableRules.ENUMERABLE_FILTER_TO_CALC_RULE);
        planner.addRule(EnumerableRules.ENUMERABLE_TABLE_SCAN_RULE);
        planner.addRule(EnumerableRules.ENUMERABLE_VALUES_RULE);
        planner.addRule(PelagoProjectPushBelowUnpack.INSTANCE());
        planner.addRule(CoreRules.PROJECT_VALUES_MERGE);
        planner.addRule(CoreRules.PROJECT_MERGE);
        planner.addRule(new RelOptRule(operand(Aggregate.class, operand(RelNode.class, operand(RelNode.class, any()))), "PelagoNoInputAggregate") {
            @Override
            public boolean matches(RelOptRuleCall call) {
                Aggregate agg = call.rel(0);

                if (!agg.getGroupSet().isEmpty()) return false;

                for (var aggCall : agg.getAggCallList()) {
                    if (!aggCall.getArgList().isEmpty()) return false;
                    if (aggCall.filterArg > 0) return false;
                }

                return (call.rel(2) instanceof PelagoTableScan)
                    && (call.rel(1) instanceof PelagoUnpack);
            }

            @Override
            public void onMatch(RelOptRuleCall call) {
                Aggregate       agg  = call.rel(0);
                PelagoTableScan scan = call.rel(2);

                var nscan =
                    PelagoTableScan.create(
                        scan.getCluster(),
                        scan.getTable(),
                        scan.pelagoTable(),
                        new int[]{}
                    );

                var nnscan = call.getPlanner().register(nscan, null);

                RelNode in = call.getPlanner().register(PelagoUnpack.create(nnscan, RelPacking.UnPckd()), null);

                call.transformTo(agg.copy(
                    agg.getTraitSet(),
                    List.of(in)
                ));
            }
        });
//
//        List<RelOptRule> rules = new ArrayList<RelOptRule>();
//        rules.add(new TableScanRule(PelagoRelFactories.PELAGO_BUILDER));
//        // push and merge filter rules
//        rules.add(new FilterAggregateTransposeRule(Filter.class, PelagoRelFactories.PELAGO_BUILDER, Aggregate.class));
//        rules.add(FilterProjectTransposeRule.INSTANCE);
////        rules.add(SubstitutionVisitor.FilterOnProjectRule.INSTANCE);
////        rules.add(new FilterMergeRule(PelagoRelFactories.PELAGO_BUILDER));
//        rules.add(FilterJoinRule.FILTER_ON_JOIN);
//        rules.add(FilterJoinRule.JOIN);
////      rules.add(new FilterJoinRule.FilterIntoJoinRule(true, RelFactories.LOGICAL_BUILDER, //PelagoRelFactories.PELAGO_BUILDER,
////          new FilterJoinRule.Predicate() {
////            public boolean apply(Join join, JoinRelType joinType, RexNode exp) {
////              return exp.isA(SqlKind.EQUALS);
////            }
////          }));
////      rules.add(new FilterJoinRule.JoinConditionPushRule(RelFactories.LOGICAL_BUILDER, /*PelagoRelFactories.PELAGO_BUILDER, */  new FilterJoinRule.Predicate() {
////        public boolean apply(Join join, JoinRelType joinType, RexNode exp) {
////          return exp.isA(SqlKind.EQUALS);
////        }
////      }));
//
//
////      rules.add(FilterJoinRule.FILTER_ON_JOIN);
////      rules.add(FilterJoinRule.JOIN);
//        /*push filter into the children of a join*/
//        rules.add(FilterTableScanRule.INSTANCE);
//        // push and merge projection rules
//        rules.add(ProjectRemoveRule.INSTANCE);
//        rules.add(ProjectJoinTransposeRule.INSTANCE);//new ProjectJoinTransposeRule(PushProjector.ExprCondition.TRUE, PelagoRelFactories.PELAGO_BUILDER));
////        rules.add(JoinProjectTransposeRule.BOTH_PROJECT);
//        rules.add(ProjectFilterTransposeRule.INSTANCE); //XXX causes non-termination
//        /*it is better to use filter first an then project*/
//        rules.add(ProjectTableScanRule.INSTANCE);
//        rules.add(PelagoProjectMergeRule.INSTANCE);//new ProjectMergeRule(true, PelagoRelFactories.PELAGO_BUILDER));
//        //aggregate rules
//        rules.add(AggregateRemoveRule.INSTANCE);
////        rules.add(AggregateReduceFunctionsRule.INSTANCE);
//        rules.add(new AggregateReduceFunctionsRule(operand(Aggregate.class, any()), PelagoRelFactories.PELAGO_BUILDER));
//        rules.add(AggregateJoinTransposeRule.EXTENDED);
//        rules.add(new AggregateProjectMergeRule(Aggregate.class, Project.class, PelagoRelFactories.PELAGO_BUILDER));
//        rules.add(new AggregateProjectPullUpConstantsRule(Aggregate.class,
//            Project.class, PelagoRelFactories.PELAGO_BUILDER,
//            "AggregateProjectPullUpConstantsRule"));
//        rules.add(new AggregateExpandDistinctAggregatesRule(Aggregate.class, true, PelagoRelFactories.PELAGO_BUILDER));
//        //join rules
//////                                                                                                                rules.add(JoinToMultiJoinRule.INSTANCE);
//////                                                                                                                rules.add(LoptOptimizeJoinRule.INSTANCE);
//////                                                                                                              rules.add(MultiJoinOptimizeBushyRule.INSTANCE);//,
////                                                                                                                rules.add(JoinPushThroughJoinRule.LEFT );
////                                                                                                                rules.add(JoinPushThroughJoinRule.RIGHT);
////                                                                                                                /*choose between right and left*/
////                                                                                                                rules.add(JoinPushExpressionsRule.INSTANCE);
////                                                                                                                rules.add(JoinAssociateRule.INSTANCE);
////                                                                                                                rules.add(JoinCommuteRule.INSTANCE);
//        // simplify expressions rules
//        rules.add(ReduceExpressionsRule.CALC_INSTANCE);
//        rules.add(ReduceExpressionsRule.FILTER_INSTANCE);
//        rules.add(ReduceExpressionsRule.PROJECT_INSTANCE);
//        rules.add(ReduceExpressionsRule.JOIN_INSTANCE);
//
//        // prune empty results rules
//        rules.add(PruneEmptyRules.FILTER_INSTANCE);
//        rules.add(PruneEmptyRules.PROJECT_INSTANCE);
//        rules.add(PruneEmptyRules.AGGREGATE_INSTANCE);
//        rules.add(PruneEmptyRules.JOIN_LEFT_INSTANCE);
//        rules.add(PruneEmptyRules.JOIN_RIGHT_INSTANCE);
//        rules.add(ProjectTableScanRule.INSTANCE);
//        rules.add(new AggregateProjectPullUpConstantsRule(Aggregate.class,
//            RelNode.class, PelagoRelFactories.PELAGO_BUILDER,
//            "AggregatePullUpConstantsRule"));
//        /* Sort Rules*/
//        rules.add(new SortJoinTransposeRule(Sort.class,
//            Join.class, PelagoRelFactories.PELAGO_BUILDER));
//        rules.add(new SortProjectTransposeRule(Sort.class, Project.class,
//            PelagoRelFactories.PELAGO_BUILDER, null));
//        //SortRemoveRule.INSTANCE, //Too aggressive when triggered over enumerables; always removes Sort
//        rules.add(SortUnionTransposeRule.INSTANCE);
//        /*Enumerable Rules*/
////        rules.add(EnumerableRules.ENUMERABLE_FILTER_RULE);
////        rules.add(EnumerableRules.ENUMERABLE_TABLE_SCAN_RULE);
////        rules.add(EnumerableRules.ENUMERABLE_PROJECT_RULE);
////        rules.add(EnumerableRules.ENUMERABLE_AGGREGATE_RULE);
////        rules.add(EnumerableRules.ENUMERABLE_JOIN_RULE);
////      rules.add(EnumerableRules.ENUMERABLE_MERGE_JOIN_RULE) //FIMXE: no mergejoin yet
////        rules.add(EnumerableRules.ENUMERABLE_SEMI_JOIN_RULE);
////        rules.add(EnumerableRules.ENUMERABLE_SORT_RULE);       //FIMXE: no support for SORT yet
////      rules.add(EnumerableRules.ENUMERABLE_UNION_RULE)      //FIMXE: no support for UNION yet
////      rules.add(EnumerableRules.ENUMERABLE_INTERSECT_RULE)  //FIMXE: no support for INTERSECT yet
////      rules.add(EnumerableRules.ENUMERABLE_MINUS_RULE)      //FIMXE: no support for MINUS yet
//        rules.add(EnumerableRules.ENUMERABLE_COLLECT_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_UNCOLLECT_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_CORRELATE_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_VALUES_RULE);
////      rules.add(EnumerableRules.ENUMERABLE_JOIN_RULE)
////
////      rules.add(EnumerableRules.ENUMERABLE_MERGE_JOIN_RULE)
////      rules.add(EnumerableRules.ENUMERABLE_SEMI_JOIN_RULE)
////      rules.add(EnumerableRules.ENUMERABLE_CORRELATE_RULE)
////      rules.add(EnumerableRules.ENUMERABLE_PROJECT_RULE)
////      rules.add(EnumerableRules.ENUMERABLE_FILTER_RULE)
////      rules.add(EnumerableRules.ENUMERABLE_AGGREGATE_RULE);
////      rules.add(EnumerableRules.ENUMERABLE_SORT_RULE);
////      rules.add(EnumerableRules.ENUMERABLE_LIMIT_RULE)
//        rules.add(EnumerableRules.ENUMERABLE_UNION_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_INTERSECT_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_MINUS_RULE);
//        rules.add(EnumerableRules.ENUMERABLE_TABLE_MODIFICATION_RULE);
////      rules.add(EnumerableRules.ENUMERABLE_VALUES_RULE)
////      rules.add(EnumerableRules.ENUMERABLE_WINDOW_RULE)
//
//        for (RelOptRule r: rules){
//            planner.addRule(r);
//        }
//
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
        cluster.setMetadataProvider(PelagoRelMetadataProvider.INSTANCE());
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
        final List<SqlOperatorTable> list = new ArrayList<>();
        list.add(opTab0);
        list.add(catalogReader);
        final SqlOperatorTable opTab = SqlOperatorTables.chain(list);
        final JavaTypeFactory typeFactory = context.getTypeFactory();
        final CalciteConnectionConfig connectionConfig = context.config();
        final SqlValidator.Config config = SqlValidator.Config.DEFAULT
            .withLenientOperatorLookup(connectionConfig.lenientOperatorLookup())
            .withSqlConformance(connectionConfig.conformance())
            .withDefaultNullCollation(connectionConfig.defaultNullCollation())
            .withIdentifierExpansion(true);
        return new CalciteSqlValidator(opTab, catalogReader, typeFactory,
            config);
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
        return Object.class.getName();
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
    public void executeDdl(Context context, SqlNode node) {
        if (node.getKind().belongsTo(Arrays.asList(SqlKind.SET_OPTION))){
            if (node instanceof SqlSetOption){
                SqlSetOption setOption = (SqlSetOption) node;
                switch (setOption.getName().names.get(0)){
                    case "compute_units":
                    case "compute":
                    case "hwmode":
                    case "cu": {
                        SqlNode value = setOption.getValue();
                        String option;
                        if (value instanceof SqlIdentifier) {
                            SqlIdentifier id = (SqlIdentifier) value;
                            option = id.toString();
                        } else if (value instanceof SqlLiteral) {
                            SqlLiteral lit = (SqlLiteral) value;
                            option = lit.toValue();
                        } else {
                            throw new UnsupportedOperationException();
                        }
                        option = option.toLowerCase();
                        switch(option){
                            case "cpu":
                            case "cpuonly":{
                                Repl.setCpuonly();
                                return;
                            }
                            case "gpu":
                            case "gpuonly":{
                                Repl.setGpuonly();
                                return;
                            }
                            case "all":
                            case "hybrid":{
                                Repl.setHybrid();
                                return;
                            }
                            default:
                                throw new UnsupportedOperationException();
                        }
                    }
                    case "cpudop": {
                        SqlNode value = setOption.getValue();
                        if (value instanceof SqlLiteral) {
                            SqlLiteral lit = (SqlLiteral) value;
                            int new_cpudop = lit.intValue(true);
                            if (new_cpudop > 0) {
                                Repl.cpudop_$eq(new_cpudop);
                                return;
                            }
                        }
                        throw new UnsupportedOperationException();
                    }
                    case "gpudop": {
                        SqlNode value = setOption.getValue();
                        if (value instanceof SqlLiteral) {
                            SqlLiteral lit = (SqlLiteral) value;
                            int new_gpudop = lit.intValue(true);
                            if (new_gpudop >= 0) {
                                Repl.gpudop_$eq(new_gpudop);
                                return;
                            }
                        }
                        throw new UnsupportedOperationException();
                    }
                    case "timings": {
                        SqlNode value = setOption.getValue();
                        String option;
                        if (value instanceof SqlIdentifier) {
                            SqlIdentifier id = (SqlIdentifier) value;
                            option = id.toString();
                        } else if (value instanceof SqlLiteral) {
                            SqlLiteral lit = (SqlLiteral) value;
                            option = lit.toValue();
                        } else {
                            throw new UnsupportedOperationException();
                        }
                        option = option.toLowerCase();
                        switch (option){
                            case "csv":{
                                Repl.timingscsv_$eq(true);
                                Repl.timings_$eq(true);
                                return;
                            }
                            case "text":
                            case "on":{
                                Repl.timingscsv_$eq(false);
                                Repl.timings_$eq(true);
                                return;
                            }
                            case "off":{
                                Repl.timings_$eq(false);
                                return;
                            }
                        }
                        throw new UnsupportedOperationException();
                    }
                }
            }
        }
        super.executeDdl(context, node);
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
                        context.getRootSchema(), prefer, createCluster(planner, new RexBuilder(typeFactory)), resultConvention,
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

                if (sqlNode.isA(Set.of(SqlKind.CREATE_TABLE))){
                    assert(sqlNode instanceof SqlCreatePelagoTable);
                    var n = ((SqlCreatePelagoTable) sqlNode).query;
                    if (n != null){
                        var name = ((SqlCreatePelagoTable) sqlNode).name;
                        SqlNode query1;
                        try {
                            final FrameworkConfig conf = Frameworks.newConfigBuilder()
                                .defaultSchema(context.getRootSchema().plus())
                                .build();
                            final Planner pl = Frameworks.getPlanner(conf);
                            final StringBuilder buf = new StringBuilder();
                            final SqlWriterConfig writerConfig =
                                SqlPrettyWriter.config().withAlwaysUseParentheses(false);
                            final SqlPrettyWriter w = new SqlPrettyWriter(writerConfig, buf);
                            buf.append("INSERT INTO ");
                            name.unparse(w, 0, 0);
                            buf.append(' ');
                            n.unparse(w, 0, 0);
                            final String sql = buf.toString();
                            query1 = pl.parse(sql);
                        } catch (SqlParseException e) {
                            throw new RuntimeException(e);
                        }

                        final SqlValidator validator =
                            createSqlValidator(context, catalogReader);

                        preparedResult = preparingStmt.prepareSql(
                            query1, Object.class, validator, true);

                        x = RelOptUtil.createDmlRowType(SqlKind.INSERT, typeFactory);
                        System.out.println(x);
                        RelDataType jdbcType = makeStruct(typeFactory, x);
                        System.out.println(jdbcType);
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
//                        assert(bindable != null);
//                        assert(context.getDataContext() != null);
//                        var xi = bindable.bind(context.getDataContext());
//
//                        assert(xi != null);
//
//                        xi.forEach(System.out::println);
//                        assert(false);
                        return new CalciteSignature<>(query1.toString(),
                            ImmutableList.of(),
                            ImmutableMap.of(), jdbcType,
                            columns, cursorFactory,
                            context.getRootSchema(), ImmutableList.of(), 1,
                            bindable,
                            getStatementType(SqlKind.INSERT));
                    }
                }

                return new CalciteSignature<>(query.sql,
                    ImmutableList.of(),
                    ImmutableMap.of(), null,
                    ImmutableList.of(), Meta.CursorFactory.OBJECT,
                    null, ImmutableList.of(), -1,
                    dataContext -> Linq4j.emptyEnumerable(),
                    Meta.StatementType.OTHER_DDL);
            }

            final SqlValidator validator =
                    createSqlValidator(context, catalogReader);

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
