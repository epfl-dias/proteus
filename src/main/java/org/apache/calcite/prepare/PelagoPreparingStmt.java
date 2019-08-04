package org.apache.calcite.prepare;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

import org.apache.calcite.adapter.enumerable.EnumerableRel;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.linq4j.function.Function2;
import org.apache.calcite.plan.*;
import org.apache.calcite.plan.hep.HepMatchOrder;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.plan.volcano.AbstractConverter;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.core.RelFactories;
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider;
import org.apache.calcite.rel.rules.FilterJoinRule;
import org.apache.calcite.rel.rules.JoinAssociateRule;
import org.apache.calcite.rel.rules.JoinCommuteRule;
import org.apache.calcite.rel.rules.JoinProjectTransposeRule;
import org.apache.calcite.rel.rules.JoinToMultiJoinRule;
import org.apache.calcite.rel.rules.ProjectJoinTransposeRule;
import org.apache.calcite.rel.rules.ProjectMergeRule;
import org.apache.calcite.rel.rules.ProjectMultiJoinMergeRule;
import org.apache.calcite.rel.rules.ProjectRemoveRule;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.runtime.Hook;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql2rel.RelDecorrelator;
import org.apache.calcite.sql2rel.RelFieldTrimmer;
import org.apache.calcite.sql2rel.SqlRexConvertletTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.Program;
import org.apache.calcite.tools.Programs;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.Holder;

import ch.epfl.dias.calcite.adapter.pelago.PelagoRelFactories;
import ch.epfl.dias.calcite.adapter.pelago.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.RelHomDistribution;
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataProvider;
import ch.epfl.dias.calcite.adapter.pelago.rules.LikeToJoinRule;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPackTransfers;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPushDeviceCrossDown;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPushDeviceCrossNSplitDown;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPushRouterDown;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoPushSplitDown;
import ch.epfl.dias.calcite.adapter.pelago.rules.PelagoRules;
import ch.epfl.dias.repl.Repl;

import java.util.List;
import java.util.Map;

import ch.epfl.dias.calcite.adapter.pelago.reporting.PelagoTimeInterval;
import ch.epfl.dias.calcite.adapter.pelago.reporting.TimeKeeper;

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

    private static class DeLikeProgram implements Program {
        public RelNode run(RelOptPlanner p, RelNode rel,
            RelTraitSet requiredOutputTraits,
            List<RelOptMaterialization> materializations,
            List<RelOptLattice> lattices) {

            HepProgram program = HepProgram.builder()
                .addRuleInstance(LikeToJoinRule.INSTANCE)
                .build();

            HepPlanner planner = new HepPlanner(
                program,
                p.getContext(),
                true,
                new Function2<RelNode, RelNode, Void>() {
                    public Void apply(RelNode oldNode, RelNode newNode) {
                        return null;
                    }
                },
                RelOptCostImpl.FACTORY);

            planner.setRoot(rel);
            return planner.findBestExp();
        }
    }

    protected RelTraitSet getDesiredRootTraitSet(RelRoot root) {//this.resultConvention
        return root.rel.getTraitSet()
            .replace(this.resultConvention)
//            .replace(PelagoRel.CONVENTION) //this.resultConvention)
            .replace(root.collation)
            .replace(RelHomDistribution.SINGLE)
            .replace(RelDeviceType.X86_64)
            .replace(RelComputeDevice.X86_64NVPTX)
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
            System.out.println(RelOptUtil.toString(rel, SqlExplainLevel.EXPPLAN_ATTRIBUTES));
            return rel;
        }
    }

    /** Program that does time measurement between pairs invocations with same PelagoTimeInterval object */
    private static class PelagoTimer implements Program {
        private PelagoTimeInterval tm;
        private String message;

        public PelagoTimer(PelagoTimeInterval tm, String message) {
            this.tm = tm;
            this.message = message;
        }

        @Override
        public RelNode run(RelOptPlanner planner, RelNode rel,
                           RelTraitSet requiredOutputTraits,
                           List<RelOptMaterialization> materializations,
                           List<RelOptLattice> lattices) {
            if(!tm.isStarted()){
                tm.start();
            } else {
                tm.stop();
                TimeKeeper.INSTANCE.addTplanning(tm);
                System.out.println(message + tm.getDifferenceMilli() + "ms");
            }
            TimeKeeper.INSTANCE.addTimestamp();
            return rel;
        }
    }

    /** Timed sequence - helper class for timedSequence method */
    private static class PelagoTimedSequence implements Program {
        private final ImmutableList<Program> programs;
        private final PelagoTimeInterval timer;

        PelagoTimedSequence(String message, Program... programs) {
            timer = new PelagoTimeInterval();

            PelagoTimer startTimer = new PelagoTimer(timer, message);
            PelagoTimer endTimer = new PelagoTimer(timer, message);

            this.programs = new ImmutableList.Builder<Program>().add(startTimer).add(programs).add(endTimer).build();
        }

        public RelNode run(RelOptPlanner planner, RelNode rel,
                           RelTraitSet requiredOutputTraits,
                           List<RelOptMaterialization> materializations,
                           List<RelOptLattice> lattices) {
            for (Program program : programs) {
                rel = program.run(planner, rel, requiredOutputTraits, materializations, lattices);
            }
            return rel;
        }
    }

    private Program timedSequence(String message, Program... programs) {
        return new PelagoTimedSequence(message, programs);
    }

    protected Program getProgram() {
        // Allow a test to override the default program.
        final Holder<Program> holder = Holder.of(null);
        Hook.PROGRAM.run(holder);
        if (holder.get() != null) {
            return holder.get();
        }

        PelagoTimeInterval tm = new PelagoTimeInterval();

        boolean cpu_only = Repl.isCpuonly();
        int     cpudop   = Repl.cpudop();
        boolean hybrid   = Repl.isHybrid();

        ImmutableList.Builder<RelOptRule> hetRuleBuilder = ImmutableList.builder();

        hetRuleBuilder.add(PelagoRules.RULES);

        if (!cpu_only) hetRuleBuilder.add(PelagoPushDeviceCrossDown.RULES);
        if (hybrid) hetRuleBuilder.add(PelagoPushDeviceCrossNSplitDown.RULES);

        if (!(cpu_only && cpudop == 1)) hetRuleBuilder.add(PelagoPushRouterDown.RULES);
        if (hybrid) hetRuleBuilder.add(PelagoPushSplitDown.RULES);

        hetRuleBuilder.add(PelagoPackTransfers.RULES);

        hetRuleBuilder.add(AbstractConverter.ExpandConversionRule.INSTANCE);

        HepProgram hepCleanUpProjects = new HepProgramBuilder()
            .addRuleInstance(ProjectMergeRule.INSTANCE)
            .addRuleInstance(JoinProjectTransposeRule.BOTH_PROJECT)
            .addRuleInstance(JoinProjectTransposeRule.LEFT_PROJECT)
            .addRuleInstance(JoinProjectTransposeRule.RIGHT_PROJECT)
            .addRuleInstance(JoinProjectTransposeRule.BOTH_PROJECT)
            .addRuleInstance(ProjectMergeRule.INSTANCE)
//                .addRuleInstance(ProjectJoinTransposeRule.INSTANCE)
            .addRuleInstance(ProjectMultiJoinMergeRule.INSTANCE).build();

        return Programs.sequence(timedSequence("Optimization time: ",
                Programs.subQuery(PelagoRelMetadataProvider.INSTANCE),
                new DecorrelateProgram(),
                new TrimFieldsProgram(),
                new PelagoProgram(),
                new DeLikeProgram(),
                new PelagoProgram(),
                Programs.of(hepCleanUpProjects, false, PelagoRelMetadataProvider.INSTANCE),
                new PelagoProgram(),
                Programs.heuristicJoinOrder(planner.getRules(), false, 2), //planner.getRules()
                new PelagoProgram(),
                Programs.ofRules(hetRuleBuilder.build()),
                new PelagoProgram()
                ));
    }
}
