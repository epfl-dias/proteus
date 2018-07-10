package org.apache.calcite.prepare;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.calcite.adapter.enumerable.EnumerableCalc;
import org.apache.calcite.adapter.enumerable.EnumerableInterpretable;
import org.apache.calcite.adapter.enumerable.EnumerableRel;
import org.apache.calcite.avatica.Meta;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.interpreter.BindableConvention;
import org.apache.calcite.interpreter.Interpreters;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.plan.*;
import org.apache.calcite.rel.RelCollation;
import org.apache.calcite.rel.RelDistribution;
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider;
import org.apache.calcite.rel.metadata.RelMetadataProvider;
import org.apache.calcite.rel.rules.JoinCommuteRule;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexProgram;
import org.apache.calcite.runtime.Bindable;
import org.apache.calcite.runtime.Hook;
import org.apache.calcite.runtime.Typed;
import org.apache.calcite.sql.SqlExplain;
import org.apache.calcite.sql.SqlExplainFormat;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlNode;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.RelDecorrelator;
import org.apache.calcite.sql2rel.RelFieldTrimmer;
import org.apache.calcite.sql2rel.SqlRexConvertletTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.Program;
import org.apache.calcite.tools.Programs;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.Holder;
import org.apache.calcite.util.Pair;

import ch.epfl.dias.calcite.adapter.pelago.PelagoRel;
import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataProvider;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class PelagoPreparingStmt extends CalcitePrepareImpl.CalcitePreparingStmt {
    private final EnumerableRel.Prefer prefer;
    private final Map<String, Object> internalParameters =
            Maps.newLinkedHashMap();

    PelagoPreparingStmt(PelagoPrepareImpl prepare,
                         CalcitePrepare.Context context,
                         CatalogReader catalogReader,
                         RelDataTypeFactory typeFactory,
                         CalciteSchema schema,
                         EnumerableRel.Prefer prefer,
                         RelOptPlanner planner,
                         Convention resultConvention,
                         SqlRexConvertletTable convertletTable) {
        super(prepare, context, catalogReader, typeFactory, schema, prefer, planner, resultConvention, convertletTable);
        this.prefer = prefer;
    }

    public Map<String, Object> getInternalParameters() {
        return internalParameters;
    }

    /** Program that de-correlates a query.
     *
     * <p>To work around
     * <a href="https://issues.apache.org/jira/browse/CALCITE-842">[CALCITE-842]
     * Decorrelator gets field offsets confused if fields have been trimmed</a>,
     * disable field-trimming in {@link SqlToRelConverter}, and run
     * {@link Programs.TrimFieldsProgram} after this program. */
    private static class DecorrelateProgram implements Program {
        public RelNode run(RelOptPlanner planner, RelNode rel,
                           RelTraitSet requiredOutputTraits,
                           List<RelOptMaterialization> materializations,
                           List<RelOptLattice> lattices) {
            final CalciteConnectionConfig config =
                    planner.getContext().unwrap(CalciteConnectionConfig.class);
            if (config != null && config.forceDecorrelate()) {
                return RelDecorrelator.decorrelateQuery(rel);
            }
            return rel;
        }
    }

    protected RelTraitSet getDesiredRootTraitSet(RelRoot root) {//this.resultConvention
        return root.rel.getTraitSet()
            .replace(this.resultConvention)
//            .replace(PelagoRel.CONVENTION) //this.resultConvention)
            .replace(root.collation)
            .replace(RelDistributions.ANY)
            .replace(RelDeviceType.X86_64)
            .simplify();
//        return root.rel.getTraitSet().replace(this.resultConvention).replace(root.collation).replace(RelDeviceType.X86_64).simplify();
    }

    /** Program that trims fields. */
    private static class TrimFieldsProgram implements Program {
        public RelNode run(RelOptPlanner planner, RelNode rel,
                           RelTraitSet requiredOutputTraits,
                           List<RelOptMaterialization> materializations,
                           List<RelOptLattice> lattices) {
            final RelBuilder relBuilder =
                RelFactories.LOGICAL_BUILDER.create(rel.getCluster(), null);
            return new RelFieldTrimmer(null, relBuilder).trim(rel);
        }
    }


    /** Program that trims fields. */
    private static class PelagoProgram implements Program {
        public RelNode run(RelOptPlanner planner, RelNode rel,
            RelTraitSet requiredOutputTraits,
            List<RelOptMaterialization> materializations,
            List<RelOptLattice> lattices) {
            System.out.println(RelOptUtil.toString(rel, SqlExplainLevel.ALL_ATTRIBUTES));
            return rel;
        }
    }

    @Override
    protected SqlToRelConverter getSqlToRelConverter(
        SqlValidator validator,
        CatalogReader catalogReader,
        SqlToRelConverter.Config config) {
//        SqlToRelConverter.Config hijacked_config = SqlToRelConverter.configBuilder().withConfig(config).withRelBuilderFactory(PelagoRelFactories.PELAGO_BUILDER).build();
        final RelOptCluster cluster = prepare.createCluster(planner, rexBuilder);
        return new SqlToRelConverter(this, validator, catalogReader, cluster,
            convertletTable, config);
    }

    protected Program getProgram() {
        // Allow a test to override the default program.
        final Holder<Program> holder = Holder.of(null);
        Hook.PROGRAM.run(holder);
        if (holder.get() != null) {
            return holder.get();
        }
        return Programs.sequence(
                Programs.subQuery(PelagoRelMetadataProvider.INSTANCE),
                new DecorrelateProgram(),
                new TrimFieldsProgram(),
                Programs.heuristicJoinOrder(planner.getRules(), false, 2),
                new PelagoProgram()

                // Second planner pass to do physical "tweaks". This the first time that
                // EnumerableCalcRel is introduced.
//                calc(metadataProvider)
        );
    }


    @Override protected PreparedResult implement(RelRoot root) {
        RelDataType resultType = root.rel.getRowType();
        boolean isDml = root.kind.belongsTo(SqlKind.DML);
        final Bindable bindable;
        if (resultConvention == BindableConvention.INSTANCE) {
            bindable = Interpreters.bindable(root.rel);
        } else {
            EnumerableRel enumerable = (EnumerableRel) root.rel;
            if (!root.isRefTrivial()) {
                final List<RexNode> projects = new ArrayList<>();
                final RexBuilder rexBuilder = enumerable.getCluster().getRexBuilder();
                for (int field : Pair.left(root.fields)) {
                    projects.add(rexBuilder.makeInputRef(enumerable, field));
                }
                RexProgram program = RexProgram.create(enumerable.getRowType(),
                        projects, null, root.validatedRowType, rexBuilder);
                enumerable = EnumerableCalc.create(enumerable, program);
            }

            try {
                CatalogReader.THREAD_LOCAL.set(catalogReader);
                bindable = EnumerableInterpretable.toBindable(internalParameters,
                        context.spark(), enumerable, prefer);
            } finally {
                CatalogReader.THREAD_LOCAL.remove();
            }
        }

        if (timingTracer != null) {
            timingTracer.traceTime("end codegen");
        }

        if (timingTracer != null) {
            timingTracer.traceTime("end compilation");
        }

        return new PreparedResultImpl(
                resultType,
                parameterRowType,
                fieldOrigins,
                root.collation.getFieldCollations().isEmpty()
                        ? ImmutableList.<RelCollation>of()
                        : ImmutableList.of(root.collation),
                root.rel,
                mapTableModOp(isDml, root.kind),
                isDml) {
            public String getCode() {
                throw new UnsupportedOperationException();
            }

            public Bindable getBindable(Meta.CursorFactory cursorFactory) {
                return bindable;
            }

            public Type getElementType() {
                return ((Typed) bindable).getElementType();
            }
        };
    }


    public PreparedResult prepareSql(
        SqlNode sqlQuery,
        SqlNode sqlNodeOriginal,
        Class runtimeContextClass,
        SqlValidator validator,
        boolean needsValidation) {
        init(runtimeContextClass);

        final SqlToRelConverter.ConfigBuilder builder =
            SqlToRelConverter.configBuilder()
                .withTrimUnusedFields(true)
                .withExpand(THREAD_EXPAND.get())
                .withExplain(sqlQuery.getKind() == SqlKind.EXPLAIN);
//                .withRelBuilderFactory(PelagoRelFactories.PELAGO_BUILDER);
        final SqlToRelConverter sqlToRelConverter =
            getSqlToRelConverter(validator, catalogReader, builder.build());

        SqlExplain sqlExplain = null;
        if (sqlQuery.getKind() == SqlKind.EXPLAIN) {
            // dig out the underlying SQL statement
            sqlExplain = (SqlExplain) sqlQuery;
            sqlQuery = sqlExplain.getExplicandum();
            sqlToRelConverter.setDynamicParamCountInExplain(
                sqlExplain.getDynamicParamCount());
        }

        RelRoot root =
            sqlToRelConverter.convertQuery(sqlQuery, needsValidation, true);
        Hook.CONVERTED.run(root.rel);

        if (timingTracer != null) {
            timingTracer.traceTime("end sql2rel");
        }

        final RelDataType resultType = validator.getValidatedNodeType(sqlQuery);
        fieldOrigins = validator.getFieldOrigins(sqlQuery);
        assert fieldOrigins.size() == resultType.getFieldCount();

        parameterRowType = validator.getParameterRowType(sqlQuery);

        // Display logical plans before view expansion, plugging in physical
        // storage and decorrelation
        if (sqlExplain != null) {
            SqlExplain.Depth explainDepth = sqlExplain.getDepth();
            SqlExplainFormat format = sqlExplain.getFormat();
            SqlExplainLevel detailLevel = sqlExplain.getDetailLevel();
            switch (explainDepth) {
            case TYPE:
                return createPreparedExplanation(resultType, parameterRowType, null,
                    format, detailLevel);
            case LOGICAL:
                return createPreparedExplanation(null, parameterRowType, root, format,
                    detailLevel);
            default:
            }
        }

        // Structured type flattening, view expansion, and plugging in physical
        // storage.
        root = root.withRel(flattenTypes(root.rel, true));

        if (this.context.config().forceDecorrelate()) {
            // Sub-query decorrelation.
            root = root.withRel(decorrelate(sqlToRelConverter, sqlQuery, root.rel));
        }

        // Trim unused fields.
        root = trimUnusedFields(root);

        Hook.TRIMMED.run(root.rel);

        // Display physical plan after decorrelation.
        if (sqlExplain != null) {
            switch (sqlExplain.getDepth()) {
            case PHYSICAL:
            default:
                root = optimize(root, getMaterializations(), getLattices());
                return createPreparedExplanation(null, parameterRowType, root,
                    sqlExplain.getFormat(), sqlExplain.getDetailLevel());
            }
        }

        root = optimize(root, getMaterializations(), getLattices());

        if (timingTracer != null) {
            timingTracer.traceTime("end optimization");
        }

        // For transformation from DML -> DML, use result of rewrite
        // (e.g. UPDATE -> MERGE).  For anything else (e.g. CALL -> SELECT),
        // use original kind.
        if (!root.kind.belongsTo(SqlKind.DML)) {
            root = root.withKind(sqlNodeOriginal.getKind());
        }
        return implement(root);
    }

}