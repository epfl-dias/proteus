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
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql.SqlKind;
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
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;

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
            .replace(RelDistributions.SINGLETON)
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

    protected Program getProgram() {
        // Allow a test to override the default program.
        final Holder<Program> holder = Holder.of(null);
        Hook.PROGRAM.run(holder);
        if (holder.get() != null) {
            return holder.get();
        }
        return Programs.sequence(
                Programs.subQuery(DefaultRelMetadataProvider.INSTANCE),
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
}