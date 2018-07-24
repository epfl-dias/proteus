package org.apache.calcite.prepare;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

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
import org.apache.calcite.rel.RelDistributions;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.logical.LogicalProject;
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
import org.apache.commons.lang.ArrayUtils;

import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataProvider;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPackTransfers;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPushDeviceCrossDown;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPushRouterDown;
import ch.epfl.dias.repl.Repl;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import ch.epfl.dias.calcite.adapter.pelago.reporting.PelagoTimeInterval;
import ch.epfl.dias.calcite.adapter.pelago.reporting.TimeKeeper;

public class                                                                                                                                                                                                                                        PelagoPreparingStmt extends CalcitePrepareImpl.CalcitePreparingStmt {
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

    /** Program that does time measurement between pairs invocations with same PelagoTimeInterval object */
    private static class PelagoTimer implements Program {
        private PelagoTimeInterval tm;
        private String message;

        public PelagoTimer(PelagoTimeInterval tm) {
            this.tm = tm;
            this.message = "Time difference: ";
        }

        public PelagoTimer(PelagoTimeInterval tm, String message) {
            this.tm = tm;
            this.message = message;
        }

        @Override
        public RelNode run(RelOptPlanner planner, RelNode rel,
                           RelTraitSet requiredOutputTraits,
                           List<RelOptMaterialization> materializations,
                           List<RelOptLattice> lattices) {

            if(!tm.getStarted()){
                tm.start();
            } else {
                tm.stop();
                TimeKeeper.getInstance().addTcalcite(tm.getDifferenceMilli());
                TimeKeeper.getInstance().addTimestamp();
                System.out.println(message + tm.getDifferenceMilli() + "ms");
            }

            return rel;
        }
    }

    /** Timed sequence - helper class for timedSequence method */
    private static class PelagoTimedSequence implements Program {
        private final ImmutableList<Program> programs;
        private final PelagoTimeInterval timer;

        PelagoTimedSequence(Program... programs) {
            timer = new PelagoTimeInterval();

            PelagoTimer startTimer = new PelagoTimer(timer);
            PelagoTimer endTimer = new PelagoTimer(timer);

            this.programs =  new ImmutableList.Builder<Program>().add(startTimer).addAll(ImmutableList.copyOf(programs)).add(endTimer).build();

        }

        PelagoTimedSequence(String message, Program... programs) {
            timer = new PelagoTimeInterval();

            PelagoTimer startTimer = new PelagoTimer(timer, message);
            PelagoTimer endTimer = new PelagoTimer(timer, message);

            this.programs =  new ImmutableList.Builder<Program>().add(startTimer).addAll(ImmutableList.copyOf(programs)).add(endTimer).build();

        }

        public RelNode run(RelOptPlanner planner, RelNode rel,
                           RelTraitSet requiredOutputTraits,
                           List<RelOptMaterialization> materializations,
                           List<RelOptLattice> lattices) {
            for (Program program : programs) {
                rel = program.run(
                        planner, rel, requiredOutputTraits, materializations, lattices);
            }
            return rel;
        }
    }

    /**  */
    private Program timedSequence(Program... programs) {
        return new PelagoTimedSequence(programs);
    }

    /** */
    private Program timedSequence(String message, Program... programs) {
        return new PelagoTimedSequence(message, programs);
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

        PelagoTimeInterval tm = new PelagoTimeInterval();

        boolean cpu_only = Repl.cpuonly();

        ImmutableList.Builder<RelOptRule> hetRuleBuilder = ImmutableList.builder();

        if (!cpu_only) hetRuleBuilder.add(PelagoPushDeviceCrossDown.RULES);

        hetRuleBuilder.add(PelagoPushRouterDown.RULES);
        hetRuleBuilder.add(PelagoPackTransfers.RULES );

        return Programs.sequence(timedSequence("Calcite time: ",
//                new PelagoProjectRootProgram(),
                Programs.subQuery(PelagoRelMetadataProvider.INSTANCE),
                new DecorrelateProgram(),
                new TrimFieldsProgram(),
                Programs.heuristicJoinOrder(planner.getRules(), false, 2),
                new PelagoProgram(),
                Programs.ofRules(hetRuleBuilder.build()),
//                Programs.ofRules(PelagoPushDeviceCrossDown.RULES),
//                new PelagoProgram(),
//                Programs.ofRules(PelagoPushRouterDown.RULES),
                new PelagoProgram()
                ));

        // Second planner pass to do physical "tweaks". This the first time that
                // EnumerableCalcRel is introduced.
//                calc(metadataProvider)
    }


    @Override protected PreparedResult implement(RelRoot root) {
        RelDataType resultType = root.rel.getRowType();
        boolean isDml = root.kind.belongsTo(SqlKind.DML);
        final Bindable bindable;
        if (resultConvention == BindableConvention.INSTANCE) {
            bindable = Interpreters.bindable(root.rel);
        } else {
            EnumerableRel enumerable = (EnumerableRel) root.rel;
//            if (!root.isRefTrivial()) {
//                final List<RexNode> projects = new ArrayList<>();
//                final RexBuilder rexBuilder = enumerable.getCluster().getRexBuilder();
//                for (int field : Pair.left(root.fields)) {
//                    projects.add(rexBuilder.makeInputRef(enumerable, field));
//                }
//                RexProgram program = RexProgram.create(enumerable.getRowType(),
//                        projects, null, root.validatedRowType, rexBuilder);
//                enumerable = EnumerableCalc.create(enumerable, program);
//            }

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

    protected RelRoot optimize(RelRoot root,
        final List<Materialization> materializations,
        final List<CalciteSchema.LatticeEntry> lattices) {
        if (!root.isRefTrivial()) {
            final List<RexNode> projects = new ArrayList<>();
            final RexBuilder rexBuilder = root.rel.getCluster().getRexBuilder();
            for (int field : Pair.left(root.fields)) {
                projects.add(rexBuilder.makeInputRef(root.rel, field));
            }
            root = root.withRel(LogicalProject.create(root.rel, projects, root.validatedRowType));
        }
        return super.optimize(root, materializations, lattices);
    }

}