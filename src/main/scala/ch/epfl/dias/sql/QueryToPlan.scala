package ch.epfl.dias.sql

import java.util

import org.apache.calcite.adapter.enumerable.EnumerableRules
import org.apache.calcite.config.Lex
import org.apache.calcite.plan._
import org.apache.calcite.prepare.CalcitePrepareImpl
import org.apache.calcite.rel.{RelCollationTraitDef, RelDistributionTraitDef, RelDistributions, RelNode}
import org.apache.calcite.rel.rules._
import org.apache.calcite.rel.`type`.RelDataTypeSystem
import org.apache.calcite.tools.Program
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.sql.fun.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParser
import org.apache.calcite.tools.FrameworkConfig
import org.apache.calcite.tools.Frameworks
import org.apache.calcite.tools.Planner
import org.apache.calcite.tools.Programs
import org.apache.calcite.tools.RuleSets
import org.apache.calcite.rel.metadata.DefaultRelMetadataProvider

import scala.collection.JavaConversions._
import org.apache.calcite.config.CalciteConnectionConfig
import org.apache.calcite.sql2rel.RelDecorrelator
import org.apache.calcite.rel.core.RelFactories
import org.apache.calcite.sql2rel.RelFieldTrimmer

/**
  * Utility that produces a query plan corresponding to an input SQL query
  *
  * Inputs:
  *  -> a schema spec file ('model')
  *  -> a folder containing a file for each relevant table. Each file contains the table schema
  *  -> a query to be parsed, validated, and optimized
  *
  *  Code primarily based on
  *   -> https://github.com/milinda/samza-2015-fork/blob/rb34664/samza-sql-calcite/src/main/java/org/apache/samza/sql/planner/QueryPlanner.java
  *   -> https://github.com/giwrgostheod/Calcite-Saber/blob/master/src/main/java/calcite/Tester.java
  */
class QueryToPlan(schema: SchemaPlus) {

    val config : FrameworkConfig = {

      //seems they are not necessary
      val traitDefs = new util.ArrayList[RelTraitDef[_ <: RelTrait]]
      traitDefs.add(ConventionTraitDef.INSTANCE)
      traitDefs.add(RelCollationTraitDef.INSTANCE)
      traitDefs.add(RelDistributionTraitDef.INSTANCE)

      val rules = new util.ArrayList[RelOptRule]
      rules.add(ReduceExpressionsRule.CALC_INSTANCE)
      rules.add(TableScanRule.INSTANCE)
      // push and merge filter rules
      rules.add(FilterAggregateTransposeRule.INSTANCE)
      rules.add(FilterProjectTransposeRule.INSTANCE)
      rules.add(FilterMergeRule.INSTANCE)
      rules.add(FilterJoinRule.FILTER_ON_JOIN)
      rules.add(FilterJoinRule.JOIN)
      /*push filter into the children of a join*/
      rules.add(FilterTableScanRule.INSTANCE)
      // push and merge projection rules
      rules.add(ProjectRemoveRule.INSTANCE)
      rules.add(ProjectJoinTransposeRule.INSTANCE)
      // rules.add(JoinProjectTransposeRule.BOTH_PROJECT)
      rules.add(ProjectFilterTransposeRule.INSTANCE) //XXX causes non-termination
      /*it is better to use filter first an then project*/
      rules.add(ProjectTableScanRule.INSTANCE)
      rules.add(ProjectMergeRule.INSTANCE)
      //aggregate rules
      rules.add(AggregateRemoveRule.INSTANCE)
      rules.add(AggregateJoinTransposeRule.INSTANCE)
      rules.add(AggregateProjectMergeRule.INSTANCE)
      rules.add(AggregateProjectPullUpConstantsRule.INSTANCE)
      rules.add(AggregateExpandDistinctAggregatesRule.INSTANCE)
//    rules.add(AggregateReduceFunctionsRule.INSTANCE) //optimizes out required sorting in some cases!
       //join rules
      rules.add(JoinToMultiJoinRule.INSTANCE)
      rules.add(LoptOptimizeJoinRule.INSTANCE)
       //        MultiJoinOptimizeBushyRule.INSTANCE,
      rules.add(JoinPushThroughJoinRule.RIGHT)
      rules.add(JoinPushThroughJoinRule.LEFT)
      /*choose between right and left*/
      rules.add(JoinPushExpressionsRule.INSTANCE)
      rules.add(JoinAssociateRule.INSTANCE)
      rules.add(JoinCommuteRule.INSTANCE)
      // simplify expressions rules
      rules.add(ReduceExpressionsRule.CALC_INSTANCE)
      rules.add(ReduceExpressionsRule.FILTER_INSTANCE)
      rules.add(ReduceExpressionsRule.PROJECT_INSTANCE)
      // prune empty results rules
      rules.add(PruneEmptyRules.FILTER_INSTANCE)
      rules.add(PruneEmptyRules.PROJECT_INSTANCE)
      rules.add(PruneEmptyRules.AGGREGATE_INSTANCE)
      rules.add(PruneEmptyRules.JOIN_LEFT_INSTANCE)
      rules.add(PruneEmptyRules.JOIN_RIGHT_INSTANCE)
      /* Sort Rules*/
      rules.add(SortJoinTransposeRule.INSTANCE)
      rules.add(SortProjectTransposeRule.INSTANCE)
      //SortRemoveRule.INSTANCE, //Too aggressive when triggered over enumerables; always removes Sort
      rules.add(SortUnionTransposeRule.INSTANCE)
      /*Enumerable Rules*/
      rules.add(EnumerableRules.ENUMERABLE_FILTER_RULE)
      rules.add(EnumerableRules.ENUMERABLE_TABLE_SCAN_RULE)
      rules.add(EnumerableRules.ENUMERABLE_PROJECT_RULE)
      rules.add(EnumerableRules.ENUMERABLE_AGGREGATE_RULE)
      rules.add(EnumerableRules.ENUMERABLE_JOIN_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_MERGE_JOIN_RULE) //FIMXE: no mergejoin yet
      rules.add(EnumerableRules.ENUMERABLE_SEMI_JOIN_RULE)
      rules.add(EnumerableRules.ENUMERABLE_SORT_RULE)       //FIMXE: no support for SORT yet
//      rules.add(EnumerableRules.ENUMERABLE_UNION_RULE)      //FIMXE: no support for UNION yet
//      rules.add(EnumerableRules.ENUMERABLE_INTERSECT_RULE)  //FIMXE: no support for INTERSECT yet
//      rules.add(EnumerableRules.ENUMERABLE_MINUS_RULE)      //FIMXE: no support for MINUS yet
      rules.add(EnumerableRules.ENUMERABLE_COLLECT_RULE)
      rules.add(EnumerableRules.ENUMERABLE_UNCOLLECT_RULE)
      rules.add(EnumerableRules.ENUMERABLE_CORRELATE_RULE)
      rules.add(EnumerableRules.ENUMERABLE_VALUES_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_JOIN_RULE)
      
//      rules.add(EnumerableRules.ENUMERABLE_MERGE_JOIN_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_SEMI_JOIN_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_CORRELATE_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_PROJECT_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_FILTER_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_AGGREGATE_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_SORT_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_LIMIT_RULE)
      rules.add(EnumerableRules.ENUMERABLE_UNION_RULE)
      rules.add(EnumerableRules.ENUMERABLE_INTERSECT_RULE)
      rules.add(EnumerableRules.ENUMERABLE_MINUS_RULE)
      rules.add(EnumerableRules.ENUMERABLE_TABLE_MODIFICATION_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_VALUES_RULE)
//      rules.add(EnumerableRules.ENUMERABLE_WINDOW_RULE)

//      val program1 = new Program {
//        def run(
//            planner: RelOptPlanner,
//            rel: RelNode,
//            requiredOutputTraits: RelTraitSet,
//            materializations: util.List[RelOptMaterialization],
//            lattices: util.List[RelOptLattice]
//          ): RelNode = {
//          planner.setRoot(rel);
//
//          for (materialization <- materializations) {
//            planner.addMaterialization(materialization);
//          }
//          for (lattice <- lattices) {
//            planner.addLattice(lattice);
//          }
//
//          System.out.println(rel.getTraitSet());
//          System.out.println(requiredOutputTraits);
//
//          val rootRel2: RelNode =
//              if (rel.getTraitSet().equals(requiredOutputTraits)) rel
//              else planner.changeTraits(rel, requiredOutputTraits);
//
//          planner.setRoot(rootRel2);
//          planner.chooseDelegate().findBestExp()
//        }
//      }
//
//      val decorrelateProgram = new Program {
//        def run(
//            planner: RelOptPlanner,
//            rel: RelNode,
//            requiredOutputTraits: RelTraitSet,
//            materializations: util.List[RelOptMaterialization],
//            lattices: util.List[RelOptLattice]
//          ): RelNode = {
//          val config = planner.getContext().unwrap(classOf[CalciteConnectionConfig])
//          if (config != null && config.forceDecorrelate())
//            RelDecorrelator.decorrelateQuery(rel)
//          else
//            rel
//        }
//      }
//
//      val trimFieldsProgram = new Program {
//        def run(
//            planner: RelOptPlanner,
//            rel: RelNode,
//            requiredOutputTraits: RelTraitSet,
//            materializations: util.List[RelOptMaterialization],
//            lattices: util.List[RelOptLattice]
//          ): RelNode = {
//          val relBuilder = PelagoRelFactories.PELAGO_BUILDER.create(rel.getCluster(), null);
//          new RelFieldTrimmer(null, relBuilder).trim(rel);
//        }
//      }


      //Can mix & match (& add)
      // Sequence and programs used are based on Calcite's Programs.standard(): 
      // https://github.com/apache/calcite/blob/be2fe5f95827eb911c49887882268749b45e372b/core/src/main/java/org/apache/calcite/tools/Programs.java
      val program: Program = Programs.sequence(
//        Programs.subQuery(DefaultRelMetadataProvider.INSTANCE),
//        decorrelateProgram, // new DecorrelateProgram(),
//        trimFieldsProgram, // new TrimFieldsProgram(),
        Programs.heuristicJoinOrder(rules, false, 3)
//        program1,

        // // Second planner pass to do physical "tweaks". This the first time that
        // // EnumerableCalcRel is introduced.
        // // Programs.calc(DefaultRelMetadataProvider.INSTANCE)
        // Programs.hep(rules, false, DefaultRelMetadataProvider.INSTANCE) // false => not DAG
      ) //Programs.standard() //.hep(rules, false, DefaultRelMetadataProvider.INSTANCE)

      Frameworks.newConfigBuilder()
        .parserConfig(SqlParser.configBuilder().setLex(Lex.MYSQL).build())
        .defaultSchema(schema)
        .operatorTable(SqlStdOperatorTable.instance())
//        .traitDefs(traitDefs)
        .context(Contexts.EMPTY_CONTEXT)
        // Rule sets to use in transformation phases.
        // Each transformation phase can use a different set of rules.
        .ruleSets(RuleSets.ofList())
        //If null, use the default cost factory for that planner.
        .costFactory(null)
        .typeSystem(RelDataTypeSystem.DEFAULT)
        //override default set of rules
        .programs(program)
        .build()

    }

    val planner : Planner = Frameworks.getPlanner(config)

    def getLogicalPlan(query: String) : RelNode = {
      val sqlNode = planner.parse(query)
      val validatedSqlNode = planner.validate(sqlNode)
      planner.rel(validatedSqlNode).project
    }

}
//select count(*) from ssbm_lineorder, ssbm_date where lo_orderdate = d_datekey and d_year = 1997