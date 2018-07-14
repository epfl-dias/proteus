package ch.epfl.dias.calcite.adapter.pelago.rules;

import ch.epfl.dias.calcite.adapter.pelago.*;
//import ch.epfl.dias.calcite.adapter.pelago.trait.RelDeviceType;
//import ch.epfl.dias.calcite.adapter.pelago.trait.RelDeviceTypeTraitDef;
import com.google.common.base.Predicate;
import com.google.common.base.Predicates;
import com.google.common.base.Supplier;

import org.apache.calcite.plan.*;
import org.apache.calcite.rel.*;
import org.apache.calcite.rel.convert.ConverterRule;
import org.apache.calcite.rel.core.*;
import org.apache.calcite.rel.logical.*;
import org.apache.calcite.rel.metadata.RelMdDistribution;
import org.apache.calcite.rel.rules.JoinCommuteRule;
import org.apache.calcite.rex.*;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.tools.RelBuilder.AggCall;

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
        PelagoPackingConverterRule.TO_PACKED_INSTANCE,
        PelagoPackingConverterRule.TO_UNPCKD_INSTANCE,
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
    };

    public static final RelOptRule[] RULES2 = {
//        PelagoDistributionConverterRule.BRDCST_INSTANCE,
        PelagoDistributionConverterRule.BRDCST_INSTANCE2,
//        PelagoDistributionConverterRule.SEQNTL_INSTANCE,
        PelagoDistributionConverterRule.SEQNTL_INSTANCE2,
        PelagoDistributionConverterRule.RANDOM_INSTANCE,

        PelagoDeviceTypeConverterRule.TO_NVPTX_INSTANCE,
        PelagoDeviceTypeConverterRule.TO_x86_64_INSTANCE,

//        PelagoProjectPushBelowDeviceCross.INSTANCE,
//        PelagoSortPushBelowDeviceCross.INSTANCE,
//        PelagoAggregatePushBelowDeviceCross.INSTANCE,
//        PelagoJoinPushBelowDeviceCross.INSTANCE,
//        PelagoFilterPushBelowDeviceCross.INSTANCE,
//        PelagoProjectPushBelowRouter.INSTANCE,
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
                if (converted != null) {
                    call.transformTo(converted);

                    // exclude unconverted node from the search space
                    // this is a good idea only if _all_ the non-converter rules
                    // operate over the {@link org.apache.calcite.rel.core} nodes (and not the logical ones)
                    call.getPlanner().setImportance(rel, 0);
                    // Without the above line, the planner was giving a higher priority to the logical nodes
                    // rather than the pelago nodes. This was resulting in an (almost) exhaustive search over
                    // all the logical plans, before moving to the pelago nodes, resulting in huge planning times.
                    // Thankfully, after we convert a logical node into a pelago node, we should not care about the
                    // logical one, as long as all the (general, non-converter) optimization rules reference the
                    // core nodes instead of the logical ones. If this is not the case, we should revisit the above
                    // line.
                }
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
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return RelDeviceType.X86_64;//.SINGLETON;
                    }
                })
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.SINGLETON;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                })
                ;

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
            RelTraitSet traitSet = agg.getInput().getTraitSet().replace(PelagoRel.CONVENTION)//rel.getCluster().traitSet()
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return RelDeviceType.X86_64;
                    }
                })
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.SINGLETON;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                })
                ;

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
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return RelDeviceType.X86_64;
                    }
                })
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.SINGLETON;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                })
                ;

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
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return RelDeviceType.X86_64;
                    }
                })
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.SINGLETON;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                })
                ;

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
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalAggregate}
     * to a {@link PelagoRouter}.
     */
    private static class PelagoRouterRule extends PelagoConverterRule {
        private static final PelagoRouterRule INSTANCE = new PelagoRouterRule();

        private PelagoRouterRule() {
            super(LogicalExchange.class, "PelagoRouterRule");
        }

        @Override public boolean matches(RelOptRuleCall call) {
            //Do we have any limitations for the reduce?
            return true;
        }

        public RelNode convert(RelNode rel) {
            final LogicalExchange xchange = (LogicalExchange) rel;
            //convert(xchange.getInput(), RelDeviceType.x86_64)
            return PelagoRouter.create(convert(xchange.getInput(), out), xchange.getDistribution());
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
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return RelDeviceType.X86_64;//.SINGLETON;
                    }
                })
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.SINGLETON;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                })
                ;
            return PelagoFilter.create(convert(filter.getInput(), traitSet), filter.getCondition());
        }
    }

    private static class PelagoJoinRuleHash extends PelagoConverterRule {
        private static final PelagoJoinRuleHash INSTANCE = new PelagoJoinRuleHash();

        private PelagoJoinRuleHash(){
            this("PelagoJoinRuleHash");
        }

        protected PelagoJoinRuleHash(String description) {
            super(LogicalJoin.class, description);
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

        public RelNode convert(RelNode rel) {
            final Join join = (Join) rel;

            JoinInfo inf = join.analyzeCondition();
            assert inf.isEqui();

            RexNode condition = join.getCondition();
            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);

            condition = disjunctions.get(0);

            RelDistribution rdl = RelDistributions.hash(inf.leftKeys );
            RelDistribution rdr = RelDistributions.hash(inf.rightKeys);

//                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
//                public RelPacking get() {
//                    return RelPacking.UnPckd;
//                }
//            })
            RelNode left  = convert(convert(join.getLeft (), out), rdl);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getLeft (), rdl, true);
            RelNode right = convert(convert(join.getRight(), out), rdr);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getRight(), rdr, true);

            return PelagoJoin.create(
                    left                  ,
                    right                 ,
                    condition             ,
                    join.getVariablesSet(),
                    join.getJoinType()
            );
        }
    }

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
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.RANDOM_DISTRIBUTED;
                    }
                }).replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return RelDeviceType.NVPTX;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                })
                ;

            RelTraitSet rightTraitSet = rel.getCluster().traitSet().replace(PelagoRel.CONVENTION)
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.BROADCAST_DISTRIBUTED;
                    }
                }).replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return RelDeviceType.NVPTX;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                })
                ;

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

    private static class PelagoJoinRule extends PelagoConverterRule {
        private static final PelagoJoinRule SEQUENTIAL_NVPTX = new PelagoJoinRule(RelDeviceType.NVPTX , RelDeviceType.NVPTX , RelDistributions.SINGLETON, RelDistributions.SINGLETON, "PelagoJoinRuleSeqNVPTX");
        private static final PelagoJoinRule SEQUENTIAL_X8664 = new PelagoJoinRule(RelDeviceType.X86_64, RelDeviceType.X86_64, RelDistributions.SINGLETON, RelDistributions.SINGLETON, "PelagoJoinRuleSeqX8664");
        private static final PelagoJoinRule BROADCAST_NVPTX  = new PelagoJoinRule(RelDeviceType.NVPTX , RelDeviceType.NVPTX , RelDistributions.BROADCAST_DISTRIBUTED, RelDistributions.RANDOM_DISTRIBUTED, "PelagoJoinRuleBrdNVPTX");
        private static final PelagoJoinRule BROADCAST_X8664  = new PelagoJoinRule(RelDeviceType.X86_64, RelDeviceType.X86_64, RelDistributions.BROADCAST_DISTRIBUTED, RelDistributions.RANDOM_DISTRIBUTED, "PelagoJoinRuleBrdX8664");

        private final RelDeviceType   leftDeviceType   ;
        private final RelDeviceType   rightDeviceType  ;
        private final RelDistribution leftDistribution ;
        private final RelDistribution rightDistribution;

        protected PelagoJoinRule(   RelDeviceType   leftDeviceType    ,
                                    RelDeviceType   rightDeviceType   ,
                                    RelDistribution leftDistribution  ,
                                    RelDistribution rightDistribution ,
                                    String          description       ){
            super(LogicalJoin.class, description);
            this.leftDeviceType    = leftDeviceType   ;
            this.rightDeviceType   = rightDeviceType  ;
            this.leftDistribution  = leftDistribution ;
            this.rightDistribution = rightDistribution;
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

//            if (join.getRight().getTraitSet().getTrait(RelDistributionTraitDef.INSTANCE) != rightDistribution) return false;
//            if (join.getLeft() .getTraitSet().getTrait(RelDistributionTraitDef.INSTANCE) != leftDistribution ) return false;
            return true;
        }

        public RelNode convert(RelNode rel) {
            Join join = (Join) rel;

            JoinInfo inf = join.analyzeCondition();
            assert inf.isEqui();

            RexNode condition = join.getCondition();
            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);

            condition = disjunctions.get(0);

            RelTraitSet leftTraitSet = rel.getCluster().traitSet().replace(out)
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return leftDistribution;
                    }
                })
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return leftDeviceType;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                });

            RelTraitSet rightTraitSet = rel.getCluster().traitSet().replace(out)
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return rightDistribution;
                    }
                })
                .replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return rightDeviceType;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                });

            RelNode left  = convert(join.getLeft (), leftTraitSet );//
            RelNode right = convert(join.getRight(), rightTraitSet);//
//            RelNode right = convert(join.getRight(), rightTraitSet);

            return PelagoJoin.create(
                left,
                right,
                condition,
                join.getVariablesSet(),
                join.getJoinType()
            );
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
        private final RelDistribution leftDistribution  = RelDistributions.SINGLETON;
        private final RelDistribution rightDistribution = RelDistributions.SINGLETON;

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

//            if (condition.isAlwaysTrue()) return false;

            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
            if (disjunctions.size() != 1)  return false;

            // Check that all conjunctions are equalities (only hashjoin supported)
            condition = disjunctions.get(0);

            for (RexNode predicate : RelOptUtil.conjunctions(condition)) {
                if (!predicate.isA(SqlKind.EQUALS)) return false;
            }

            return true;
        }

        public RelNode convert(RelNode rel) {
            Join join = (Join) rel;

            JoinInfo inf = join.analyzeCondition();
            assert inf.isEqui();

            RexNode condition = join.getCondition();
            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);

            condition = disjunctions.get(0);

            RelTraitSet leftTraitSet = rel.getCluster().traitSet().replace(out)
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return leftDistribution;
                    }
                }).replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return leftDeviceType  ;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                });

            RelTraitSet rightTraitSet = rel.getCluster().traitSet().replace(out)
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return rightDistribution;
                    }
                }).replaceIf(RelDeviceTypeTraitDef.INSTANCE, new Supplier<RelDeviceType>() {
                    public RelDeviceType get() {
                        return rightDeviceType  ;
                    }
                })
                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
                    public RelPacking get() {
                        return RelPacking.UnPckd;
                    }
                });

            RelNode left  = convert(join.getLeft (), leftTraitSet );
            RelNode right = convert(join.getRight(), rightTraitSet);

            join = PelagoJoin.create(
                left                  ,
                right                 ,
                condition             ,
                join.getVariablesSet(),
                join.getJoinType()
            );

            final RelNode  swapped = (swap) ? JoinCommuteRule.swap(join, false) : join;
            if (swapped == null) return null;

//            rel.getCluster().getPlanner().setImportance(rel, 0);
            return swapped;
        }
    }

    private static class PelagoJoinRule3 extends PelagoConverterRule {
        private static final PelagoJoinRule3 INSTANCE = new PelagoJoinRule3();

        private PelagoJoinRule3(){
            this("PelagoJoinRule3");
        }

        protected PelagoJoinRule3(String description) {
            super(LogicalJoin.class, description);
        }

        @Override
        public boolean matches(RelOptRuleCall call) {
            final LogicalJoin join = (LogicalJoin) call.rel(0);
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

        public RelNode convert(RelNode rel) {
            final LogicalJoin join = (LogicalJoin) rel;

            JoinInfo inf = join.analyzeCondition();
            assert inf.isEqui();

            RexNode condition = join.getCondition();
            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);

            condition = disjunctions.get(0);

//            RelDistribution rdl = RelDistributions.hash(inf.leftKeys );
//            RelDistribution rdr = RelDistributions.hash(inf.rightKeys);

//                .replaceIf(RelPackingTraitDef     .INSTANCE, new Supplier<RelPacking     >() {
//                public RelPacking get() {
//                    return RelPacking.UnPckd;
//                }
//            })
//            RelNode left  = convert(convert(join.getLeft (), out), rdl);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getLeft (), rdl, true);
            RelNode left  = convert(convert(join.getLeft (), out), RelDistributions.BROADCAST_DISTRIBUTED);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getLeft (), rdl, true);
//            RelNode right = convert(convert(join.getRight(), out), rdr);//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getRight(), rdr, true);
            RelNode right = convert(convert(join.getRight(), out), RelDistributions.RANDOM_DISTRIBUTED   );//RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getRight(), rdr, true);

//            final RelTraitSet traitSet = join.getTraitSet(); //Both rdl and rdr, can we propagate this information ?

            return PelagoJoin.create(
                    left                  ,
                    right                 ,
                    condition             ,
                    join.getVariablesSet(),
                    join.getJoinType()
            );
        }
    }

//
//    private static class PelagoBroadCastJoinRule extends PelagoJoinRule {
//        private static final PelagoBroadCastJoinRule INSTANCE = new PelagoBroadCastJoinRule();
//
//        private PelagoBroadCastJoinRule() {
//            this("PelagoBroadCastJoinRule");
//        }
//
//        private PelagoBroadCastJoinRule(String description) {
//            super(description);
//        }
//
//        public RelNode convert(RelNode rel) {
//            final LogicalJoin join = (LogicalJoin) rel;
//            RexNode condition = join.getCondition();
////            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
////
////            condition = disjunctions.get(0);
//
////            ImmutableBitSet left__bitset = ImmutableBitSet.of();
////            ImmutableBitSet right_bitset = ImmutableBitSet.of();
////            int lcount = join.getLeft().getRowType().getFieldCount();
////            for (Integer x: RelOptUtil.InputFinder.analyze(condition).inputBitSet.build()){
////                if (x < lcount) {
////                    left__bitset = left__bitset.set(x);
////                } else {
////                    right_bitset = right_bitset.set(x - lcount);
////                }
////            }
//
//            RelNode left  = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getLeft (), RelDistributions.BROADCAST_DISTRIBUTED, true);
//            RelNode right = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getRight(), RelDistributions.RANDOM_DISTRIBUTED   , true);
//
//            final RelTraitSet traitSet = right.getTraitSet().replace(out); //.plus(RelDistributions.RANDOM_DISTRIBUTED); //copy distribution from right
//            return PelagoJoin.create(
//                    convert(left , out)   ,
//                    convert(right, out)   ,
//                    condition             ,
//                    join.getVariablesSet(),
//                    join.getJoinType()
//            );
//        }
//
//        @Override
//        public boolean matches(RelOptRuleCall call) {
//            final LogicalJoin join = (LogicalJoin) call.rel(0);
//            if (join.getLeft().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE) != join.getRight().getTraitSet().getTrait(RelDeviceTypeTraitDef.INSTANCE)) return false;
//            return super.matches(call);
////            if (!super.matches(call)) return false;
////            return ((LogicalJoin) call.rel(0)).getLeft().getTraitSet().satisfies(RelTraitSet.createEmpty().plus(RelDistributions.BROADCAST_DISTRIBUTED));
////            final LogicalJoin join = (LogicalJoin) call.rel(0);
////            RexNode condition = join.getCondition();
////
////            if (condition.isAlwaysTrue()) return false;
////
////            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
////            if (disjunctions.size() != 1)  return false;
////
////            // Check that all conjunctions are equalities (only hashjoin supported)
////            condition = disjunctions.get(0);
////
////            for (RexNode predicate : RelOptUtil.conjunctions(condition)) {
////                if (!predicate.isA(SqlKind.EQUALS)) return false;
////            }
////
////            return true;
//        }
//    }

//    private static class PelagoBroadCastJoinRule2 extends PelagoJoinRule {
//        private static final PelagoBroadCastJoinRule2 INSTANCE = new PelagoBroadCastJoinRule2();
//
//        private PelagoBroadCastJoinRule2() {
//            super("PelagoBroadCastJoinRule2");
//        }
//
//        public RelNode convert(RelNode rel) {
//            final LogicalJoin join = (LogicalJoin) rel;
//            RexNode condition = join.getCondition();
////            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
////
////            condition = disjunctions.get(0);
//
////            ImmutableBitSet left__bitset = ImmutableBitSet.of();
////            ImmutableBitSet right_bitset = ImmutableBitSet.of();
////            int lcount = join.getLeft().getRowType().getFieldCount();
////            for (Integer x: RelOptUtil.InputFinder.analyze(condition).inputBitSet.build()){
////                if (x < lcount) {
////                    left__bitset = left__bitset.set(x);
////                } else {
////                    right_bitset = right_bitset.set(x - lcount);
////                }
////            }
//
//            RelNode left  = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getLeft (), RelDistributions.RANDOM_DISTRIBUTED, true);
//            RelNode right = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join.getRight(), RelDistributions.BROADCAST_DISTRIBUTED   , true);
//
//            final RelTraitSet traitSet = left.getTraitSet().replace(out); //copy distribution from right
//            return new PelagoJoin(join.getCluster(), traitSet,
//                    convert(left , out)   ,
//                    convert(right, out)   ,
//                    condition             ,
//                    join.getVariablesSet(),
//                    join.getJoinType()
//            );
//        }
//    }

//    private static class PelagoBroadCastJoinRule2 extends PelagoBroadCastJoinRule {
//        private static final PelagoBroadCastJoinRule2 INSTANCE = new PelagoBroadCastJoinRule2();
//
//        private PelagoBroadCastJoinRule2() {
//            super("PelagoBroadCastJoinRule2");
//        }
//
//        public RelNode convert(RelNode rel) {
//            final LogicalJoin join = (LogicalJoin) rel;
//            final RelNode  swapped = JoinCommuteRule.swap(join, false);
//            if (swapped == null) return null;
//
//            return swapped;
////            // The result is either a Project or, if the project is trivial, a
////            // raw Join.
////            final Join join2 = (swapped instanceof Join) ? (Join) swapped : (Join) swapped.getInput(0);
////            RexNode condition = join.getCondition();
////
////
////
////
//////            List<RexNode> disjunctions = RelOptUtil.disjunctions(condition);
//////
//////            condition = disjunctions.get(0);
////
//////            ImmutableBitSet left__bitset = ImmutableBitSet.of();
//////            ImmutableBitSet right_bitset = ImmutableBitSet.of();
//////            int lcount = join.getLeft().getRowType().getFieldCount();
//////            for (Integer x: RelOptUtil.InputFinder.analyze(condition).inputBitSet.build()){
//////                if (x < lcount) {
//////                    left__bitset = left__bitset.set(x);
//////                } else {
//////                    right_bitset = right_bitset.set(x - lcount);
//////                }
//////            }
////
////            RelNode left  = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join2.getLeft (), RelDistributions.RANDOM_DISTRIBUTED, true);
////            RelNode right = RelDistributionTraitDef.INSTANCE.convert(rel.getCluster().getPlanner(), join2.getRight(), RelDistributions.BROADCAST_DISTRIBUTED   , true);
////
////            final RelTraitSet traitSet = left.getTraitSet().replace(out); //copy distribution from right
////            return new PelagoJoin(join2.getCluster(), traitSet,
////                    convert(left , out)   ,
////                    convert(right, out)   ,
////                    condition             ,
////                    join2.getVariablesSet(),
////                    join2.getJoinType()
////            );
//        }
//    }
}