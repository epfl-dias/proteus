package ch.epfl.dias.calcite.adapter.pelago.rules;

import ch.epfl.dias.calcite.adapter.pelago.*;

import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.rules.JoinCommuteRule;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/**
 * Rules and relational operators for
 * {@link PelagoRel#CONVENTION}
 * calling convention.
 */
public class PelagoRules {
    private PelagoRules() {
    }

    public static final RelOptRule[] RULES = {
        PelagoProjectTableScanRule.INSTANCE,
        PelagoToEnumerableConverterRule.INSTANCE,
        PelagoProjectPushBelowUnpack.INSTANCE,
        PelagoProjectRule.INSTANCE,
        PelagoAggregateRule.INSTANCE,
        PelagoSortRule.INSTANCE,
        PelagoFilterRule.INSTANCE,
        PelagoUnnestRule.INSTANCE,
//        PelagoJoinSeq.INSTANCE,
        PelagoJoinSeq.INSTANCE2, //Use the instance that swaps, as Lopt seems to generate left deep plans only
    };

    /** Base class for planner rules that convert a relational expression to
     * Pelago calling convention. */
    abstract static class PelagoConverterRule extends ConverterRule {
        protected final Convention out;

        PelagoConverterRule(Class<? extends RelNode> clazz,
                               String description) {
            super(clazz, (Predicate<RelNode>) (e) -> true, Convention.NONE, PelagoRel.CONVENTION, PelagoRelFactories.PELAGO_BUILDER, description);
            this.out = PelagoRel.CONVENTION;
        }

        public void onMatch(RelOptRuleCall call) {
            RelNode rel = call.rel(0);
            if (rel.getTraitSet().contains(Convention.NONE)) {
                final RelNode converted = convert(rel);
                if (converted != null) call.transformTo(converted);
            }
        }
    }

    /**
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalProject}
     * to a {@link PelagoProject}.
     */
    private static class PelagoProjectRule extends PelagoConverterRule {
        private static final PelagoProjectRule INSTANCE = new PelagoProjectRule();

        private PelagoProjectRule() {
            super(LogicalProject.class, "PelagoProjectRule");
        }

        @Override public boolean matches(RelOptRuleCall call) {
            return true;
        }

        public RelNode convert(RelNode rel) {
            final Project project = (Project) rel;

            RelTraitSet traitSet = project.getInput().getTraitSet().replace(out)//rel.getCluster().traitSet()
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)//.SINGLETON
                .replace(RelHomDistribution.SINGLE)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

            return PelagoProject.create(convert(project.getInput(), traitSet), project.getProjects(), project.getRowType());
        }
    }

    /**
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalAggregate}
     * to a {@link PelagoProject}.
     */
    private static class PelagoAggregateRule extends PelagoConverterRule {
        private static final PelagoAggregateRule INSTANCE = new PelagoAggregateRule();

        private PelagoAggregateRule() {
            super(LogicalAggregate.class, "PelagoAggregateRule");
        }

        public boolean matches(RelOptRuleCall call) {
            return true;
        }

        public RelNode convert(RelNode rel) {
            final Aggregate agg = (Aggregate) rel;

            RelTraitSet traitSet = agg.getTraitSet().replace(PelagoRel.CONVENTION)
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)
                .replace(RelHomDistribution.SINGLE)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

            RelNode inp = convert(agg.getInput(), traitSet);
            return PelagoAggregate.create(inp, agg.indicator, agg.getGroupSet(), agg.getGroupSets(), agg.getAggCallList());
        }
    }

    /**
     * Rule to create a {@link PelagoUnnest}.
     */
    private static class PelagoUnnestRule extends RelOptRule {
        private static final PelagoUnnestRule INSTANCE = new PelagoUnnestRule();

        private PelagoUnnestRule() {
            super(
                operand(
                    LogicalCorrelate.class,
                    operand(
                        RelNode.class,
                        any()
                    ),
                    operand(
                        Uncollect.class,
                        operand(
                            LogicalProject.class,
                            any()
                        )
                    )
                ),
                PelagoRelFactories.PELAGO_BUILDER,
                "PelagoUnnestRule"
            );
        }

        public boolean matches(RelOptRuleCall call) {
            return true;
        }

        @Override public void onMatch(final RelOptRuleCall call) {
            Correlate correlate = call.rel(0);
            if (!(correlate.getTraitSet().contains(Convention.NONE))) return;

            RelNode   input     = call.rel(1);
            Uncollect uncollect = call.rel(2);
            Project   proj      = call.rel(3);

            RelTraitSet traitSet = uncollect.getTraitSet()
                .replace(PelagoRel.CONVENTION)
                .replace(RelDeviceType.X86_64)
                .replace(RelHomDistribution.SINGLE)
                .replace(RelPacking.UnPckd);

            call.transformTo(
                PelagoUnnest.create(
                    convert(input, traitSet),
                    correlate.getCorrelationId(),
                    proj.getNamedProjects(),
                    uncollect.getRowType()
                )
            );
        }
    }

    /**
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalSort}
     * to a {@link PelagoSort}.
     */
    private static class PelagoSortRule extends PelagoConverterRule {
        private static final PelagoSortRule INSTANCE = new PelagoSortRule();

        private PelagoSortRule() {
            super(LogicalSort.class, "PelagoSortRule");
        }

        public boolean matches(RelOptRuleCall call) {
            return true;
        }

        public RelNode convert(RelNode rel) {
            final Sort sort = (Sort) rel;

            RelTraitSet traitSet = sort.getInput().getTraitSet().replace(PelagoRel.CONVENTION)
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)
                .replace(RelHomDistribution.SINGLE)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

            RelNode inp = convert(sort.getInput(), traitSet);

           return PelagoSort.create(inp, sort.collation, sort.offset, sort.fetch);
        }
    }

    /**
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalFilter} to a
     * {@link PelagoFilter}.
     */
    private static class PelagoFilterRule extends PelagoConverterRule {
        private static final PelagoFilterRule INSTANCE = new PelagoFilterRule();

        private PelagoFilterRule() {
            super(LogicalFilter.class, "PelagoFilterRule");
        }

        @Override
        public boolean matches(RelOptRuleCall call) {
            return true;
        }

        public RelNode convert(RelNode rel) {
            final Filter filter = (Filter) rel;

            RelTraitSet traitSet = filter.getInput().getTraitSet().replace(out)//rel.getCluster().traitSet()
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)//.SINGLETON
                .replace(RelHomDistribution.SINGLE)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);
            return PelagoFilter.create(convert(filter.getInput(), traitSet), filter.getCondition());
        }
    }

    private static class PelagoJoinSeq extends PelagoConverterRule {
        private static final PelagoJoinSeq INSTANCE  = new PelagoJoinSeq("PelagoJoinSeqRule" , false);
        private static final PelagoJoinSeq INSTANCE2 = new PelagoJoinSeq("PelagoJoinSeqRule2", true );

        private PelagoJoinSeq(){
            this("PelagoJoinSeqRule");
        }

        private final RelDeviceType   leftDeviceType    = RelDeviceType.X86_64;//.NVPTX;
        private final RelDeviceType   rightDeviceType   = RelDeviceType.X86_64;//.NVPTX;
        private final RelHomDistribution leftDistribution  = RelHomDistribution.SINGLE;
        private final RelHomDistribution rightDistribution = RelHomDistribution.SINGLE;

        private final boolean swap;

        protected PelagoJoinSeq(String description) {
            this(description, false);
        }

        protected PelagoJoinSeq(String description, boolean swap) {
            super(LogicalJoin.class, description);
            this.swap = swap;
        }

        @Override
        public boolean matches(RelOptRuleCall call) {
            final Join join = (Join) call.rel(0);

//            if (join.getLeft().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE) != join.getRight().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)) return false;

            RexNode condition = join.getCondition();

            if (condition.isAlwaysTrue()) return false;

            JoinInfo inf = join.analyzeCondition();
            if (inf.isEqui()) return true;

            condition = RexUtil.toCnf(join.getCluster().getRexBuilder(), condition);
//            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
//            if (disjunctions.size() != 1)  return false;

            // Check that all conjunctions are equalities (only hashjoin supported)
//            condition = disjunctions.get(0);

            for (RexNode predicate : RelOptUtil.conjunctions(condition)) {
                if (predicate.isA(SqlKind.EQUALS)) return true;
            }

            return false;
        }

        @Override public RelNode convert(final RelNode rel) {
            assert(false) : "wrong convert called, as it needs RelOptRuleCall";
            return null;
        }

        public void onMatch(RelOptRuleCall call) {
            RelNode rel = call.rel(0);
            if (rel.getTraitSet().contains(Convention.NONE)) {
                final RelNode converted = convert(rel, call);
                if (converted != null) call.transformTo(converted);
            }
        }

        public RelNode convert(RelNode rel, RelOptRuleCall call) {
            Join join = (Join) rel;

            RexNode cond = join.getCondition();

            JoinInfo inf = join.analyzeCondition();
            List<RexNode> equalities = new ArrayList();
            List<RexNode> rest       = new ArrayList();
            List<RexNode> rest0      = new ArrayList();
            List<RexNode> rest1      = new ArrayList();
            int thr = join.getLeft().getRowType().getFieldCount();
            if (inf.isEqui()) {
                equalities.add(cond);
            } else {
                RexNode condition = RexUtil.pullFactors(join.getCluster().getRexBuilder(), cond);
                assert(condition.isA(SqlKind.AND));
                for (RexNode predicate: ((RexCall) condition).getOperands()){
                    // Needs a little bit of fixing... not completely correct checking
                    RelOptUtil.InputFinder vis = new RelOptUtil.InputFinder();
                    predicate.accept(vis);
                    boolean rel0 = false;
                    boolean rel1 = false;
                    for (int acc: RelOptUtil.InputFinder.bits(predicate)){
                        rel0 = rel0 || (acc <  thr);
                        rel1 = rel1 || (acc >= thr);
                    }
                    if (predicate.isA(SqlKind.EQUALS)) {
                        if (rel0 && rel1) {
                            equalities.add(predicate);
                            continue;
                        }
                    }

                    if (rel0 && !rel1) {
                        rest0.add(predicate);
                    } else if (!rel0 && rel1){
                        rest1.add(predicate);
                    } else {
                        rest.add(predicate);
                    }
                }
            }

            final RexBuilder rexBuilder = join.getCluster().getRexBuilder();

            RexNode joinCond  = RexUtil.composeConjunction(rexBuilder, equalities, false);
            RexNode leftCond  = RexUtil.composeConjunction(rexBuilder, rest0, false);
            RexNode rightCond = RexUtil.shift(RexUtil.composeConjunction(rexBuilder, rest1, false), -thr);
            RexNode aboveCond = RexUtil.composeConjunction(rexBuilder, rest, false);

            RelTraitSet leftTraitSet = rel.getCluster().traitSet().replace(out)
                .replace(leftDistribution)
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> leftDeviceType)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

            RelTraitSet rightTraitSet = rel.getCluster().traitSet().replace(out)
                .replace(rightDistribution)
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> rightDeviceType)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);


            RelNode preLeft  = convert(join.getLeft (), leftTraitSet );
            RelNode left     = (!rest0.isEmpty()) ? PelagoFilter.create(preLeft , leftCond ) : preLeft ;

            RelNode preRight = convert(join.getRight(), rightTraitSet);
            RelNode right    = (!rest1.isEmpty()) ? PelagoFilter.create(preRight, rightCond) : preRight;

            join = PelagoJoin.create(
                left                  ,
                right                 ,
                joinCond              ,
                join.getVariablesSet(),
                join.getJoinType()
            );

            RelNode  swapped = (swap) ? JoinCommuteRule.swap(join, false, call.builder()) : join;
            if (swapped == null) return null;

            if (swap){
                final Join newJoin =
                    swapped instanceof Join
                        ? (Join) swapped
                        : (Join) swapped.getInput(0);

                final RelBuilder relBuilder = call.builder();
                final List<RexNode> exps =
                    RelOptUtil.createSwappedJoinExprs(newJoin, join, false);
                relBuilder.push(swapped)
                    .project(exps, newJoin.getRowType().getFieldNames());

                call.getPlanner().ensureRegistered(relBuilder.build(), newJoin);
            }

            swapped = convert(swapped, PelagoRel.CONVENTION);

            if (rest.isEmpty()) return swapped;


            RelNode root = PelagoFilter.create(
                swapped,
                aboveCond
            );

            rel.getCluster().getPlanner().ensureRegistered(root, join);
            return root;
        }
    }
}
