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
        PelagoDistributionConverterRule.BRDCST_INSTANCE,
        PelagoDistributionConverterRule.BRDCST_INSTANCE2,
        PelagoDistributionConverterRule.SEQNTL_INSTANCE,
        PelagoDistributionConverterRule.SEQNTL_INSTANCE2,
        PelagoDistributionConverterRule.RANDOM_INSTANCE,
        PelagoDeviceTypeConverterRule.TO_NVPTX_INSTANCE ,
        PelagoDeviceTypeConverterRule.TO_x86_64_INSTANCE,
        PelagoProjectTableScanRule.INSTANCE,
        PelagoToEnumerableConverterRule.INSTANCE,
        PelagoDeviceCrossRule.INSTANCE,
        PelagoProjectRule.INSTANCE,
        PelagoAggregateRule.INSTANCE,
        PelagoSortRule.INSTANCE,
        PelagoFilterRule.INSTANCE,
        PelagoRouterRule.INSTANCE,
        PelagoJoinRule2.INSTANCE,
//        JoinCommuteRule.INSTANCE,
//        PelagoJoinSeq.INSTANCE,
//        PelagoJoinRule.SEQUENTIAL_NVPTX,
//        PelagoJoinRule.SEQUENTIAL_X8664,
        PelagoJoinRule.BROADCAST_NVPTX ,
//        PelagoJoinRule.BROADCAST_X8664 ,
//        PelagoJoinRuleHash.INSTANCE    ,
//        PelagoJoinRule3.INSTANCE,
//        PelagoBroadCastJoinRule.INSTANCE,
//        PelagoBroadCastJoinRule2.INSTANCE
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
            super(clazz, predicate, Convention.NONE, PelagoRel.CONVENTION, RelFactories.LOGICAL_BUILDER, description);
            this.out = PelagoRel.CONVENTION;
        }
    }

    /**
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalProject}
     * to a {@link PelagoProject}.
     */
    private static class PelagoProjectRule extends PelagoConverterRule {
        private static final PelagoProjectRule INSTANCE = new PelagoProjectRule();

        private PelagoProjectRule() {
            super(Project.class, "PelagoProjectRule");
        }

        @Override public boolean matches(RelOptRuleCall call) {
            return true;
        }

        public RelNode convert(RelNode rel) {
            final Project project = (Project) rel;
            return PelagoProject.create(convert(project.getInput(), out), project.getProjects(), project.getRowType());
        }
    }

    /**
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalAggregate}
     * to a {@link PelagoProject}.
     */
    private static class PelagoAggregateRule extends PelagoConverterRule {
        private static final PelagoAggregateRule INSTANCE = new PelagoAggregateRule();

        private PelagoAggregateRule() {
            super(Aggregate.class, "PelagoAggregateRule");
        }

        public boolean matches(RelOptRuleCall call) {
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
            RelTraitSet traitSet = rel.getCluster().traitSet().replace(PelagoRel.CONVENTION)
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.SINGLETON;
                    }
                })
                ;

//            System.out.println("=====" + traitSet);
            RelNode inp = convert(convert(agg.getInput(), out), RelDistributions.SINGLETON);
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
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalSort}
     * to a {@link PelagoSort}.
     */
    private static class PelagoSortRule extends PelagoConverterRule {
        private static final PelagoSortRule INSTANCE = new PelagoSortRule();

        private PelagoSortRule() {
            super(Sort.class, "PelagoSortRule");
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
            RelTraitSet traitSet = rel.getCluster().traitSet().replace(PelagoRel.CONVENTION)
                .replaceIf(RelDistributionTraitDef.INSTANCE, new Supplier<RelDistribution>() {
                    public RelDistribution get() {
                        return RelDistributions.SINGLETON;
                    }
                })
                ;

//            System.out.println("=====" + traitSet);
            RelNode inp = convert(convert(sort.getInput(), out), RelDistributions.SINGLETON);
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
            RelNode r = PelagoSort.create(inp, sort.collation, sort.offset, sort.fetch);
//                System.out.println(r.getTraitSet());
            return r;
//            }
        }
    }

    /**
     * Rule to convert a {@link org.apache.calcite.rel.logical.LogicalAggregate}
     * to a {@link PelagoRouter}.
     */
    private static class PelagoDeviceCrossRule extends PelagoConverterRule {
        private static final PelagoDeviceCrossRule INSTANCE = new PelagoDeviceCrossRule();

        private PelagoDeviceCrossRule() {
            super(LogicalDeviceCross.class, "PelagoDeviceCrossRule");
        }

        @Override public boolean matches(RelOptRuleCall call) {
            //Do we have any limitations for the reduce?
            return true;
        }

        public RelNode convert(RelNode rel) {
            final LogicalDeviceCross xchange = (LogicalDeviceCross) rel;
            //convert(xchange.getInput(), RelDeviceType.x86_64)
            return PelagoDeviceCross.create(convert(xchange.getInput(), out), xchange.deviceType);
        }
    }

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
            final LogicalFilter filter = (LogicalFilter) rel;
            return PelagoFilter.create(convert(filter.getInput(), out), filter.getCondition());
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
            super(Join.class, description);
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
                });

            RelNode left  = convert(convert(convert(join.getLeft (), out) , leftDeviceType ), leftDistribution ); //convert(join.getLeft (), leftTraitSet );//
            RelNode right = convert(convert(convert(join.getRight(), out) , rightDeviceType), rightDistribution); //convert(join.getRight(), rightTraitSet);//
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
        private static final PelagoJoinSeq INSTANCE = new PelagoJoinSeq();

        private PelagoJoinSeq(){
            this("PelagoJoinSeqRule");
        }

        private final RelDeviceType   leftDeviceType    = RelDeviceType.NVPTX;
        private final RelDeviceType   rightDeviceType   = RelDeviceType.NVPTX;
        private final RelDistribution leftDistribution  = RelDistributions.SINGLETON;
        private final RelDistribution rightDistribution = RelDistributions.SINGLETON;

        protected PelagoJoinSeq(String description) {
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

            final RelNode  swapped = JoinCommuteRule.swap(join, false);
            if (swapped == null) return null;

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