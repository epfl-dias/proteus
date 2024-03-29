package org.apache.calcite.prepare;

import ch.epfl.dias.calcite.adapter.pelago.*;
import ch.epfl.dias.calcite.adapter.pelago.metadata.PelagoRelMetadataProvider;
import ch.epfl.dias.calcite.adapter.pelago.rel.*;
import ch.epfl.dias.calcite.adapter.pelago.reporting.PelagoTimeInterval;
import ch.epfl.dias.calcite.adapter.pelago.reporting.TimeKeeper;
import ch.epfl.dias.calcite.adapter.pelago.rules.*;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelComputeDevice;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelDeviceType;
import ch.epfl.dias.calcite.adapter.pelago.traits.RelHomDistribution;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;

import org.apache.calcite.adapter.enumerable.EnumerableRel;
import org.apache.calcite.adapter.enumerable.EnumerableRules;
import org.apache.calcite.config.CalciteConnectionConfig;
import org.apache.calcite.jdbc.CalcitePrepare;
import org.apache.calcite.jdbc.CalciteSchema;
import org.apache.calcite.plan.*;
import org.apache.calcite.plan.hep.HepMatchOrder;
import org.apache.calcite.plan.hep.HepPlanner;
import org.apache.calcite.plan.hep.HepProgram;
import org.apache.calcite.plan.hep.HepProgramBuilder;
import org.apache.calcite.plan.volcano.AbstractConverter;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.RelRoot;
import org.apache.calcite.rel.RelVisitor;
import org.apache.calcite.rel.RelWriter;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.externalize.RelWriterImpl;
import org.apache.calcite.rel.hint.HintPredicates;
import org.apache.calcite.rel.hint.HintStrategy;
import org.apache.calcite.rel.hint.HintStrategyTable;
import org.apache.calcite.rel.rules.*;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.runtime.Hook;
import org.apache.calcite.sql.SqlExplainLevel;
import org.apache.calcite.sql.validate.SqlValidator;
import org.apache.calcite.sql2rel.RelDecorrelator;
import org.apache.calcite.sql2rel.RelFieldTrimmer;
import org.apache.calcite.sql2rel.SqlRexConvertletTable;
import org.apache.calcite.sql2rel.SqlToRelConverter;
import org.apache.calcite.tools.Program;
import org.apache.calcite.tools.Programs;
import org.apache.calcite.tools.RelBuilder;
import org.apache.calcite.util.Holder;

import ch.epfl.dias.repl.Repl;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.*;

import scala.MatchError;
import scala.NotImplementedError;

public class PelagoPreparingStmt extends CalcitePrepareImpl.CalcitePreparingStmt {
    private final Map<String, Object> internalParameters =
            Maps.newLinkedHashMap();
    private final RelOptCluster cluster;

    PelagoPreparingStmt(PelagoPrepareImpl prepare,
                         CalcitePrepare.Context context,
                         CatalogReader catalogReader,
                         RelDataTypeFactory typeFactory,
                         CalciteSchema schema,
                         EnumerableRel.Prefer prefer,
                         RelOptCluster cluster,
                         Convention resultConvention,
                         SqlRexConvertletTable convertletTable) {
        super(prepare, context, catalogReader, typeFactory, schema, prefer, cluster, resultConvention, convertletTable);
        this.cluster = cluster;
    }

    public Map<String, Object> getInternalParameters() {
        return internalParameters;
    }

    @Override protected SqlToRelConverter getSqlToRelConverter(
        SqlValidator validator,
        CatalogReader catalogReader,
        SqlToRelConverter.Config config) {
        var config2 = config.withHintStrategyTable(
            HintStrategyTable.builder().hintStrategy("query_info", HintStrategy.builder(HintPredicates.or(
                HintPredicates.AGGREGATE, HintPredicates.JOIN, HintPredicates.CALC, HintPredicates.PROJECT, HintPredicates.TABLE_SCAN))
            .build()).build()
        );
        return new SqlToRelConverter(this, validator, catalogReader, cluster,
            convertletTable, config2);
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
                return RelDecorrelator.decorrelateQuery(rel,
                    RelFactories.LOGICAL_BUILDER.create(rel.getCluster(), null));
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
                .addRuleInstance(LikeToJoinRule.INSTANCE())
                .build();

            HepPlanner planner = new HepPlanner(
                program,
                p.getContext(),
                true,
                (oldNode, newNode) -> null,
                RelOptCostImpl.FACTORY);

            planner.setRoot(rel);
            return planner.findBestExp();
        }
    }

    protected RelTraitSet getDesiredRootTraitSet(RelRoot root) {//this.resultConvention
        return root.rel.getTraitSet()
            .replace(this.resultConvention)
//            .replace(PelagoRel.CONVENTION()) //this.resultConvention)
            .replace(root.collation)
            .replace(RelHomDistribution.SINGLE())
            .replace(RelDeviceType.X86_64())
            .replace(RelComputeDevice.X86_64NVPTX())
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

    /** Program that trims fields. */
    private static class PelagoProgramPrintPlan implements Program {
        public RelNode run(RelOptPlanner planner, RelNode rel,
            RelTraitSet requiredOutputTraits,
            List<RelOptMaterialization> materializations,
            List<RelOptLattice> lattices) {
            if (Repl.printplan() && rel instanceof PelagoToEnumerableConverter) {
                try {
                    ((PelagoToEnumerableConverter) rel).writePlan(((PelagoToEnumerableConverter) rel).getPlan(),
                        Repl.planfile());
                } catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("Failed to generate plan");
                }
            }
            return rel;
        }
    }

    private static class PelagoProgram2 implements Program {
        public RelNode run(RelOptPlanner planner, RelNode rel,
            RelTraitSet requiredOutputTraits,
            List<RelOptMaterialization> materializations,
            List<RelOptLattice> lattices) {
            if (Repl.printplan() && rel instanceof PelagoToEnumerableConverter) {
                var splits = new HashMap<Long, Integer>();
                new RelVisitor(){
                    public void visit(RelNode node, int ordinal, RelNode parent) {
                        if (node instanceof PelagoSplit){
                            var id = ((PelagoSplit) node).splitId();
                            splits.put(id, splits.getOrDefault(id, 0) + 1);
                        }
                        node.childrenAccept(this);
                    }
                }.go(rel);
                splits.forEach((key, value) -> {
                    System.out.println("Split" + key + ": " + value);
                    assert (value % 2 == 0);
                });

                try{
                    final StringWriter sw = new StringWriter();
                    final RelWriter planWriter =
                        new RelBuilderWriter(
                            new PrintWriter(sw), SqlExplainLevel.EXPPLAN_ATTRIBUTES, false);
                    rel.explain(planWriter);
                    try {
                        System.out.println(Repl.planfile() + ".cpp");
                        PrintWriter pw = new PrintWriter(Repl.planfile() + ".cpp");
                        pw.write("// AUTOGENERATED FILE. DO NOT EDIT.\n\n");
                        pw.write("constexpr auto query = __FILE__;\n\n");
                        pw.write("#include \"query.cpp.inc\"\n\n");
                        pw.write("\nPreparedStatement Query::prepare(bool memmv) {\n");
                        pw.write(sw.toString());
                        pw.write("}\n");
                        pw.close();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    }
                } catch (NotImplementedError | MatchError e){
//                    e.printStackTrace();
                }

                try {
                    System.out.println(Repl.planfile() + ".plan");
                    PrintWriter pw = new PrintWriter(Repl.planfile() + ".plan");
                    final RelWriter planWriter =
                        new RelWriterImpl(
                            new PrintWriter(pw), SqlExplainLevel.NO_ATTRIBUTES, false);
                    rel.explain(planWriter);
                    pw.close();
                } catch (FileNotFoundException e) {
                    e.printStackTrace();
                }
            }
            return rel;
        }
    }

    /** Program that does time measurement between pairs invocations with same PelagoTimeInterval object */
    private static class PelagoTimer implements Program {
        final private PelagoTimeInterval tm;
        final private String message;

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
                TimeKeeper.INSTANCE().addTplanning(tm);
                System.out.println(message + tm.getDifferenceMilli() + "ms");
            }
            TimeKeeper.INSTANCE().addTimestamp();
            return rel;
        }
    }

    /** Timed sequence - helper class for timedSequence method */
    private static class PelagoTimedSequence implements Program {
        private final ImmutableList<Program> programs;

        PelagoTimedSequence(String message, Program... programs) {
            PelagoTimeInterval timer = new PelagoTimeInterval();

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

        boolean cpu_only = Repl.isCpuonly();
        boolean gpu_only = Repl.isGpuonly();
        int     cpudop   = Repl.cpudop();
        int     gpudop   = Repl.gpudop();
        boolean hybrid   = Repl.isHybrid();

        ImmutableList.Builder<RelOptRule> hetRuleBuilder = ImmutableList.builder();

//        hetRuleBuilder.add(PelagoRules.RULES);

        if (!cpu_only) hetRuleBuilder.add(PelagoPushDeviceCrossDown.RULES());
//        if (hybrid) hetRuleBuilder.add(PelagoPushDeviceCrossNSplitDown.RULES);

        if (!(cpu_only && cpudop == 1) && !(gpu_only && gpudop == 1)) hetRuleBuilder.add(PelagoPushRouterDown.RULES());
        if (hybrid) hetRuleBuilder.add(PelagoPushSplitDown.RULES());
        if (hybrid) hetRuleBuilder.add(PelagoPullUnionUp.RULES());

        hetRuleBuilder.add(PelagoPackTransfers.RULES());
        hetRuleBuilder.add(PelagoPartialAggregateRule.INSTANCE());

        hetRuleBuilder.add(AbstractConverter.ExpandConversionRule.INSTANCE);
        hetRuleBuilder.add(JoinCommuteRule.Config.DEFAULT.withOperandFor(PelagoJoin.class).withRelBuilderFactory(
                RelBuilder.proto(
                    PelagoRelFactories.PELAGO_PROJECT_FACTORY()
                )
        ).toRule());
        hetRuleBuilder.add(ProjectMergeRule.Config.DEFAULT.withOperandFor(PelagoProject.class).withRelBuilderFactory(
            RelBuilder.proto(
                PelagoRelFactories.PELAGO_PROJECT_FACTORY()
            )
        ).toRule());

//        hetRuleBuilder.add(new ProjectJoinTransposeRule(
//            PelagoProject.class, PelagoJoin.class,
//            expr -> !(expr instanceof RexOver),
//            PelagoRelFactories.PELAGO_BUILDER));

        // To allow the join ordering program to proceed, we need to pull all
        // Project operators up (and anything else that is not a filter).
        // Operators between the joins (with the exception of Filters) do not
        // allow joins to be combined into a single MultiJoin and thus such
        // such operators create hard boundaries for the join ordering program.
        HepProgram hepPullUpProjects = new HepProgramBuilder()
            .addRuleInstance(CoreRules.PROJECT_MERGE)
            // Push Filters down
            .addRuleInstance(CoreRules.FILTER_PROJECT_TRANSPOSE)
            .addRuleInstance(CoreRules.FILTER_INTO_JOIN)
            // Pull Projects up
            .addRuleInstance(CoreRules.JOIN_PROJECT_BOTH_TRANSPOSE)
            .addGroupBegin()
            .addRuleInstance(CoreRules.JOIN_PROJECT_LEFT_TRANSPOSE)
            .addRuleInstance(CoreRules.JOIN_PROJECT_RIGHT_TRANSPOSE)
            .addGroupEnd()
            .addRuleInstance(PruneEmptyRules.PROJECT_INSTANCE)
            .addRuleInstance(CoreRules.PROJECT_REMOVE)
            .addRuleInstance(CoreRules.PROJECT_MERGE)
            .build();

        HepProgram hepPushDownProjects = new HepProgramBuilder()
            // Pull Filters up over projects
            .addRuleInstance(CoreRules.PROJECT_FILTER_TRANSPOSE)
            // Push Projects down
            .addRuleInstance(CoreRules.PROJECT_JOIN_TRANSPOSE)
            .addRuleInstance(PruneEmptyRules.PROJECT_INSTANCE)
            .addRuleInstance(CoreRules.PROJECT_REMOVE)
            .addRuleInstance(CoreRules.PROJECT_MERGE)
            .addRuleInstance(CoreRules.PROJECT_TABLE_SCAN)
            .addRuleInstance(PelagoProjectTableScanRule.INSTANCE())
            .addRuleInstance(PelagoProjectPushBelowUnpack.INSTANCE())
            .build();


        HepProgram hepPushDownProjects2 = new HepProgramBuilder()
            .addRuleInstance(CoreRules.PROJECT_MERGE)
            // Pull Filters up over projects
            .addRuleInstance(CoreRules.PROJECT_FILTER_TRANSPOSE)
//            .addRuleInstance(SortProjectTransposeRule.INSTANCE)
            // Push Projects down
            .addRuleInstance(ProjectJoinTransposeRule.Config.DEFAULT.withOperandFor(Project.class, PelagoLogicalJoin.class)
                .withRelBuilderFactory(
                    RelBuilder.proto(
                        (RelFactories.JoinFactory) (left, right, hints, condition, variablesSet, joinType, semiJoinDone)
                            -> new PelagoLogicalJoin(left.getCluster(), left.getTraitSet(), left, right, condition, variablesSet, joinType)
                    )
            ).toRule())
            .addRuleInstance(PruneEmptyRules.PROJECT_INSTANCE)
            .addRuleInstance(CoreRules.PROJECT_REMOVE)
            .addRuleInstance(CoreRules.PROJECT_MERGE)
            .addRuleInstance(CoreRules.PROJECT_TABLE_SCAN)
            .addRuleInstance(PelagoProjectTableScanRule.INSTANCE())
            .addRuleInstance(PelagoProjectPushBelowUnpack.INSTANCE())
            .build();

        // program1, program2 are based on Programs.heuristicJoinOrder

        // Ideally, the intermediate plan should contain a single MultiJoin
        // and no other joins/multijoins.

        // Create a program that gathers together joins as a MultiJoin.
        final HepProgram hep = new HepProgramBuilder()
            .addRuleInstance(CoreRules.FILTER_INTO_JOIN)
            .addMatchOrder(HepMatchOrder.BOTTOM_UP)
            .addRuleInstance(CoreRules.JOIN_TO_MULTI_JOIN)
//            .addRuleInstance(PelagoRules.PelagoFilterRule.INSTANCE)
            .build();
        final Program program1 =
            Programs.of(hep, false, PelagoRelMetadataProvider.INSTANCE());

        // Create a program that contains a rule to expand a MultiJoin
        // into heuristically ordered joins.
        // Do not add JoinCommuteRule and JoinPushThroughJoinRule, as
        // they cause exhaustive search.
        final Program program2 = Programs.of(new HepProgramBuilder()
            .addRuleInstance(LoptOptimizeJoinRule.Config.DEFAULT.withRelBuilderFactory(
                RelBuilder.proto(
                    (RelFactories.JoinFactory) (left, right, hints, condition, variablesSet, joinType, semiJoinDone)
                        -> new PelagoLogicalJoin(left.getCluster(), left.getTraitSet(), left, right, condition, variablesSet, joinType)
                )
            ).toRule())
            .build(), false, PelagoRelMetadataProvider.INSTANCE());

        final Program programFinalize = Programs.of(new HepProgramBuilder()
          .addRuleInstance(EnumerableRules.ENUMERABLE_FILTER_TO_CALC_RULE)
          .addRuleInstance(EnumerableRules.ENUMERABLE_PROJECT_TO_CALC_RULE)
          .build(), false, PelagoRelMetadataProvider.INSTANCE());

        HepProgram hepReduceProjects = new HepProgramBuilder()
//            .addRuleInstance(PruneEmptyRules.PROJECT_INSTANCE)
            .addMatchOrder(HepMatchOrder.BOTTOM_UP)
            // Projects in the probe side of a Join can be pulled up to avoid redundant calculation
            // Note that this relies on the compile to hoist the calculation for M:N joins,
            // which may not be happening
            // FIXME: adding the following rule is buggy as for a Join(Project, Project) it will
            //   only pull up the prob-eside project, if it pulls up the build-side one as well.
//            .addRuleInstance(new JoinProjectTransposeRule(
//                operand(
//                    PelagoJoin.class,
//                    operand(RelNode.class, any()),
//                    operand(PelagoProject.class, any())),
//                "JoinProjectTransposeRule(Other-Project)",
//                false,
//                PelagoRelFactories.PELAGO_BUILDER))
            .addRuleInstance(ProjectJoinTransposeRule.Config.DEFAULT
                .withOperandFor(PelagoProject.class, PelagoJoin.class)
                .withRelBuilderFactory(PelagoRelFactories.PELAGO_BUILDER())
                .toRule())
            .addRuleInstance(ProjectRemoveRule.Config.DEFAULT.withRelBuilderFactory(PelagoRelFactories.PELAGO_BUILDER()).toRule())
            .addRuleInstance(ProjectMergeRule.Config.DEFAULT.withForce(true).withRelBuilderFactory(PelagoRelFactories.PELAGO_BUILDER()).toRule())
            .build();

        return Programs.sequence(
            timedSequence("Optimization time: ",
                timedSequence(
                    "Subqueries: ",
                    Programs.subQuery(PelagoRelMetadataProvider.INSTANCE()),
                    new DecorrelateProgram(),
                    new TrimFieldsProgram()
                ),
                timedSequence(
                    "LIKE-to-join: ",
                    new DeLikeProgram()
                ),
                timedSequence(
                    "Project consolidation: ",
                    Programs.of(hepPushDownProjects, false, PelagoRelMetadataProvider.INSTANCE()),
                    Programs.of(hepPullUpProjects, false, PelagoRelMetadataProvider.INSTANCE())
                ),
                timedSequence(
                    "To multi-join: ",
                    program1
                ),
                timedSequence(
                    "Join ordering: ",
                    program2,
                    new PelagoProgram()
                ),
                timedSequence(
                    "Push down projects: ",
                    Programs.of(hepPushDownProjects2, false, PelagoRelMetadataProvider.INSTANCE())
//                    new PelagoProgram()
                ),
                timedSequence(
                    "Physical optimization: ",
                    Programs.ofRules(planner.getRules())
                ),
                timedSequence(
                    "Join commute: ",
                    Programs.ofRules(JoinCommuteRule.Config.DEFAULT.withOperandFor(PelagoJoin.class).toRule()),
                    new PelagoProgram()
                ),
                timedSequence(
                    "Reduce projects: ",
                    Programs.of(hepReduceProjects, false, PelagoRelMetadataProvider.INSTANCE())
//                    new PelagoProgram()
                ),
                timedSequence(
                    "Parallelization: ",
                    Programs.ofRules(hetRuleBuilder.build())
                ),
                timedSequence(
                    "Finalization: ",
                    programFinalize
                ),
                new PelagoProgram(),
                new PelagoProgram2(),
                new PelagoProgramPrintPlan()
            )
        );
    }
}
