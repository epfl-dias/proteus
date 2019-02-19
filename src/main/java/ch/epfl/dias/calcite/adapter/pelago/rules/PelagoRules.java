package ch.epfl.dias.calcite.adapter.pelago.rules;

import ch.epfl.dias.calcite.adapter.pelago.*;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;

import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.rules.JoinCommuteRule;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.SqlKind;

import java.util.ArrayList;
import java.util.List;

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
//        PelagoPackingConverterRule.TO_PACKED_INSTANCE,
//        PelagoPackingConverterRule.TO_UNPCKD_INSTANCE,
        PelagoProjectPushBelowUnpack.INSTANCE,
//        PelagoDeviceCrossRule.INSTANCE,
        PelagoProjectRule.INSTANCE,
        PelagoAggregateRule.INSTANCE,
        PelagoSortRule.INSTANCE,
        PelagoFilterRule.INSTANCE,
        PelagoUnnestRule.INSTANCE,
//        PelagoRouterRule.INSTANCE,
//                                                PelagoJoinRule2.INSTANCE,
//        JoinCommuteRule.INSTANCE,
//        PelagoJoinSeq.INSTANCE,
        PelagoJoinSeq.INSTANCE2, //Use the instance that swaps, as Lopt seems to generate left deep plans only
//        PelagoJoinRule.SEQUENTIAL_NVPTX,
//        PelagoJoinRule.SEQUENTIAL_X8664,
//                                                PelagoJoinRule.BROADCAST_NVPTX ,
//        PelagoJoinRule.BROADCAST_X8664 ,
//        PelagoJoinRuleHash.INSTANCE    ,
//        PelagoJoinRule3.INSTANCE,
//        PelagoBroadCastJoinRule.INSTANCE,
//        PelagoBroadCastJoinRule2.INSTANCE,
//        PelagoPushRouterBelowAggregate.INSTANCE,
    };

    /** Base class for planner rules that convert a relational expression to
     * Pelago calling convention. */
    abstract static class PelagoConverterRule extends ConverterRule {
        protected final Convention out;

        PelagoConverterRule(Class<? extends RelNode> clazz,
                               String description) {
            this(clazz, Predicates.<RelNode>alwaysTrue(), description);
        }

        <R extends RelNode> PelagoConverterRule(Class<R> clazz,
                                                   Predicate<? super R> predicate,
                                                   String description) {
            super(clazz, predicate, Convention.NONE, PelagoRel.CONVENTION, PelagoRelFactories.PELAGO_BUILDER, description);
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
//            for (AggregateCall agg: ((Aggregate) call.rel(0)).getAggCallList()){
//                if (agg.getAggregation().getKind() == SqlKind.AVG) return false;
//            }
            return true;
//            return call.rel(0).getTraitSet().contains(RelDistributions.SINGLETON);
        }

        public RelNode convert(RelNode rel) {
            final Aggregate agg = (Aggregate) rel;

//            RelNode inp = convert(agg.getInput(), out);
////            if (!inp.getTraitSet().satisfies(RelTraitSet.createEmpty().plus(RelDistributions.SINGLETON))){
////                inp = PelagoAggregate.create(inp, agg.indicator, agg.getGroupSet(), agg.getGroupSets(), agg.getAggCallList());
//
////                Mappings.
//
//                inp = convert(inp, RelDistributions.SINGLETON);

//            System.out.println(agg.getTraitSet());
//            RelTraitSet traitSet = agg.getInput().getTraitSet().replace(PelagoRel.CONVENTION)//rel.getCluster().traitSet()
            RelTraitSet traitSet = agg.getTraitSet().replace(PelagoRel.CONVENTION)//rel.getCluster().traitSet()
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)
                .replace(RelHomDistribution.SINGLE)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

//            System.out.println("=====" + traitSet);
            RelNode inp = convert(agg.getInput(), traitSet);//, RelDistributions.SINGLETON);
//            RelNode inp = convert(agg.getInput(), traitSet);

//                List<AggregateCall> alist = new ArrayList<AggregateCall>();
//                for (AggregateCall aggr: agg.getAggCallList()){
//                    SqlAggFunction ag = aggr.getAggregation();
////                    ag.
////                    if (ag instanceof SqlSplittableAggFunction){
//                    AggregateCall a = ag.unwrap(SqlSplittableAggFunction.class).split(aggr, Mappings.createIdentity(agg.getInput().getRowType().getFieldCount() - 1));
////                                aggr.isDistinct(),
////                                aggr.isApproximate(),
////                                aggr.getArgList(),
////                                aggr.filterArg,
////                                agg.getGroupCount(),
////                                inp,
////                                null,
////                                "");
//                     alist.add(a);
////                    } else {
////                        System.out.println("adasdasdasdasdasd" + ag);
////                        return null;
////                    }
//                }
//
//                System.out.println(agg.getAggCallList() + " " + alist);
//
//                return PelagoAggregate.create(inp, agg.indicator, agg.getGroupSet(), agg.getGroupSets(), alist);
//            } else {
                RelNode r = PelagoAggregate.create(inp, agg.indicator, agg.getGroupSet(), agg.getGroupSets(), agg.getAggCallList());
//                System.out.println(r.getTraitSet());
                return r;
//            }
        }
    }

    /**
     * Rule to create a {@link PelagoUnnest}.
     */
    private static class PelagoUnnestRule extends RelOptRule {
        private static final PelagoUnnestRule INSTANCE = new PelagoUnnestRule();

        private PelagoUnnestRule() {
            super(//LogicalCorrelate.class,
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
//            Correlate correlate = (Correlate) call.rel(0);
//
//            if (!(correlate.getRight() instanceof Uncollect)) return false;
//            Uncollect uncollect = (Uncollect) correlate.getRight();
//            return true;

//            if (!(uncollect.getInput() instanceof Project )) return false;
//            Project   proj      = (Project  ) uncollect.getInput();
//
//            if (!(proj     .getInput() instanceof Values  )) return false;
//            Values    val       = (Values   ) proj.getInput();

//            return val.tuples.size() == 1;
        }

        @Override public void onMatch(final RelOptRuleCall call) {
            RelNode rel = call.rel(0);
            if (!(rel.getTraitSet().contains(Convention.NONE))) return;
//                final RelNode converted = convert(rel);
//                if (converted != null) {
//                    call.transformTo(converted);
//                }
//            }
//        }
//
//        public RelNode convert(RelNode rel) {
            RelNode   input     =             call.rel(1);
            Correlate correlate = (Correlate) call.rel(0);
            Uncollect uncollect = (Uncollect) call.rel(2);//correlate.getRight();
            Project   proj      = (Project  ) call.rel(3);//uncollect.getInput();


            RelTraitSet traitSet = uncollect.getTraitSet().replace(PelagoRel.CONVENTION)//rel.getCluster().traitSet()
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)
                .replace(RelHomDistribution.SINGLE)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

//            System.out.println("=====" + traitSet);
            RelNode inp = convert(input, traitSet);//, RelDistributions.SINGLETON);

            call.transformTo(
                PelagoUnnest.create(
                    inp,
//                    uncollect,
                    correlate.getCorrelationId(),
//                    correlate.getRequiredColumns(),
//                    correlate.getJoinType(),
//                    correlate.getRowType()
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
//            return call.rel(0).getTraitSet().contains(RelDistributions.SINGLETON);
        }

        public RelNode convert(RelNode rel) {
            final Sort sort = (Sort) rel;


//            RelNode inp = convert(agg.getInput(), out);
////            if (!inp.getTraitSet().satisfies(RelTraitSet.createEmpty().plus(RelDistributions.SINGLETON))){
////                inp = PelagoAggregate.create(inp, agg.indicator, agg.getGroupSet(), agg.getGroupSets(), agg.getAggCallList());
//
////                Mappings.
//
//                inp = convert(inp, RelDistributions.SINGLETON);

//            System.out.println(agg.getTraitSet());
            RelTraitSet traitSet = sort.getInput().getTraitSet().replace(PelagoRel.CONVENTION)
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.X86_64)
                .replace(RelHomDistribution.SINGLE)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

//            System.out.println("=====" + traitSet);
//            RelNode inp = convert(convert(sort.getInput(), out), RelDistributions.SINGLETON);
            RelNode inp = convert(sort.getInput(), traitSet);

//                List<AggregateCall> alist = new ArrayList<AggregateCall>();
//                for (AggregateCall aggr: agg.getAggCallList()){
//                    SqlAggFunction ag = aggr.getAggregation();
////                    ag.
////                    if (ag instanceof SqlSplittableAggFunction){
//                    AggregateCall a = ag.unwrap(SqlSplittableAggFunction.class).split(aggr, Mappings.createIdentity(agg.getInput().getRowType().getFieldCount() - 1));
////                                aggr.isDistinct(),
////                                aggr.isApproximate(),
////                                aggr.getArgList(),
////                                aggr.filterArg,
////                                agg.getGroupCount(),
////                                inp,
////                                null,
////                                "");
//                     alist.add(a);
////                    } else {
////                        System.out.println("adasdasdasdasdasd" + ag);
////                        return null;
////                    }
//                }
//
//                System.out.println(agg.getAggCallList() + " " + alist);
//
//                return PelagoAggregate.create(inp, agg.indicator, agg.getGroupSet(), agg.getGroupSets(), alist);
//            } else {
            RelNode r = PelagoSort.create(inp, sort.collation, sort.offset, sort.fetch);
//                System.out.println(r.getTraitSet());
            return r;
//            }
        }
    }

//    /**
//     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalAggregate}
//     * to a {@link PelagoRouter}.
//     */
//    private static class PelagoDeviceCrossRule extends PelagoConverterRule {
//        private static final PelagoDeviceCrossRule INSTANCE = new PelagoDeviceCrossRule();
//
//        private PelagoDeviceCrossRule() {
//            super(LogicalDeviceCross.class, "PelagoDeviceCrossRule");
//        }
//
//        @Override public boolean matches(RelOptRuleCall call) {
//            //Do we have any limitations for the reduce?
//            return true;
//        }
//
//        public RelNode convert(RelNode rel) {
//            final LogicalDeviceCross xchange = (LogicalDeviceCross) rel;
//            //convert(xchange.getInput(), RelDeviceType.x86_64)
//            return PelagoDeviceCross.create(convert(xchange.getInput(), out), xchange.deviceType);
//        }
//    }

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

//    private static class PelagoJoinRuleHash extends PelagoConverterRule {
//        private static final PelagoJoinRuleHash INSTANCE = new PelagoJoinRuleHash();
//
//        private PelagoJoinRuleHash(){
//            this("PelagoJoinRuleHash");
//        }
//
//        protected PelagoJoinRuleHash(String description) {
//            super(LogicalJoin.class, description);
//        }
//
//        @Override
//        public boolean matches(RelOptRuleCall call) {
//            final Join join = (Join) call.rel(0);
//
////            if (join.getLeft().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE) != join.getRight().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)) return false;
//
//            RexNode condition = join.getCondition();
//
//            if (condition.isAlwaysTrue()) return false;
//
//            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
//            if (disjunctions.size() != 1)  return false;
//
//            // Check that all conjunctions are equalities (only hashjoin supported)
//            condition = disjunctions.get(0);
//
//            for (RexNode predicate : RelOptUtil.conjunctions(condition)) {
//                if (!predicate.isA(SqlKind.EQUALS)) return false;
//            }
//
//            return true;
//        }
//
//        public RelNode convert(RelNode rel) {
//            final Join join = (Join) rel;
//
//            JoinInfo inf = join.analyzeCondition();
//            assert inf.isEqui();
//
//            RexNode condition = join.getCondition();
//            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
//
//            condition = disjunctions.get(0);
//
//            RelDistribution rdl = RelHomDistribution.hash(inf.leftKeys );
//            RelDistribution rdr = RelHomDistribution.hash(inf.rightKeys);
//
////                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
////                public RelPacking get() {
////                    return RelPacking.UnPckd;
////                }
////            })
//            RelNode left  = convert(convert(join.getLeft (), out), rdl);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getLeft (), rdl, true);
//            RelNode right = convert(convert(join.getRight(), out), rdr);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getRight(), rdr, true);
//
//            return PelagoJoin.create(
//                    left                  ,
//                    right                 ,
//                    condition             ,
//                    join.getVariablesSet(),
//                    join.getJoinType()
//            );
//        }
//    }

    private static class PelagoJoinRule2 extends RelOptRule {
        private static final PelagoJoinRule2 INSTANCE = new PelagoJoinRule2();

        private PelagoJoinRule2(){
            this("PelagoJoinRule2");
        }

        protected PelagoJoinRule2(String description) {
            super(operand(LogicalJoin.class, any()), description);
        }

        @Override
        public boolean matches(RelOptRuleCall call) {
            final Join join = (Join) call.rel(0);

//            if (join.getLeft().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE) != join.getRight().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)) return false;

            RexNode condition = join.getCondition();

            if (condition.isAlwaysTrue()) return false;

            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
            if (disjunctions.size() != 1)  return false;

            // Check that all conjunctions are equalities (only hashjoin supported)
            condition = disjunctions.get(0);

            for (RexNode predicate : RelOptUtil.conjunctions(condition)) {
                if (!predicate.isA(SqlKind.EQUALS)) return false;
            }

            return true;
        }

        public void onMatch(RelOptRuleCall call) {
            RelNode rel = call.rel(0);
//            if (rel.getTraitSet().contains(this.inTrait)) {
                RelNode converted = this.convert(rel);
                if (converted != null) {
                    call.transformTo(converted);
                }
//            }
        }

        public RelNode convert(RelNode rel) {
            Join join = (Join) rel;

            JoinInfo inf = join.analyzeCondition();
            assert inf.isEqui();

            RexNode condition = join.getCondition();
            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);

            condition = disjunctions.get(0);

//            RelDistribution rdl = RelDistributions.hash(inf.leftKeys );
//            RelDistribution rdr = RelDistributions.hash(inf.rightKeys);

//            RelNode left  = convert(convert(join.getLeft (), out), RelDistributions.RANDOM_DISTRIBUTED   );//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getLeft (), rdl, true);
//            RelNode right = convert(convert(join.getRight(), out), RelDistributions.BROADCAST_DISTRIBUTED);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getRight(), rdr, true);

            RelTraitSet leftTraitSet = rel.getCluster().traitSet().replace(PelagoRel.CONVENTION)
                .replace(RelHomDistribution.RANDOM)
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.NVPTX)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

            RelTraitSet rightTraitSet = rel.getCluster().traitSet().replace(PelagoRel.CONVENTION)
                .replace(RelHomDistribution.BRDCST)
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, () -> RelDeviceType.NVPTX)
                .replaceIf(RelPackingTraitDef.INSTANCE, () -> RelPacking.UnPckd);

            RelNode left  = convert(join.getLeft (), leftTraitSet );//convert(convert(convert(join.getLeft (), out) , leftDeviceType ), leftDistribution );
            RelNode right = convert(join.getRight(), rightTraitSet);//convert(convert(convert(join.getRight(), out) , rightDeviceType), rightDistribution);
//            RelNode right = convert(join.getRight(), rightTraitSet);

            join = PelagoJoin.create(
                    left                  ,
                    right                 ,
                    condition             ,
                    join.getVariablesSet(),
                    join.getJoinType()
            );

            final RelNode swapped = JoinCommuteRule.swap(join, false);
//            if (swapped == null) return null;
            return swapped;
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

        public RelNode convert(RelNode rel) {
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

            RelNode  swapped = (swap) ? JoinCommuteRule.swap(join, false) : join;
            if (swapped == null) return null;
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
