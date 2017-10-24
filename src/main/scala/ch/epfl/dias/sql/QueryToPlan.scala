package ch.epfl.dias.sql

import java.util

import org.apache.calcite.adapter.enumerable.EnumerableRules
import org.apache.calcite.config.Lex
import org.apache.calcite.plan._
import org.apache.calcite.prepare.CalcitePrepareImpl
import org.apache.calcite.rel.RelCollationTraitDef
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.rules._
import org.apache.calcite.rel.`type`.RelDataTypeSystem
import org.apache.calcite.tools.Program
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.sql.fun.SqlStdOperatorTable
import org.apache.calcite.sql.parser.SqlParseException
import org.apache.calcite.sql.parser.SqlParser
import org.apache.calcite.tools.FrameworkConfig
import org.apache.calcite.tools.Frameworks
import org.apache.calcite.tools.Planner
import org.apache.calcite.tools.Programs
import org.apache.calcite.tools.RuleSets

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


      //Can mix & match (& add)
      //XXX Some combinations lead to a non-terminating QO process
      val program: Program = Programs.ofRules(
        ReduceExpressionsRule.CALC_INSTANCE,
        TableScanRule.INSTANCE,
        // push and merge filter rules
        FilterAggregateTransposeRule.INSTANCE,
        FilterProjectTransposeRule.INSTANCE,
        FilterMergeRule.INSTANCE,
        FilterJoinRule.FILTER_ON_JOIN,
        FilterJoinRule.JOIN,
        /*push filter into the children of a join*/
        FilterTableScanRule.INSTANCE,
        // push and merge projection rules
        ProjectRemoveRule.INSTANCE,
        ProjectJoinTransposeRule.INSTANCE,
        JoinProjectTransposeRule.BOTH_PROJECT,
//      ProjectFilterTransposeRule.INSTANCE, //XXX causes non-termination
//      /*it is better to use filter first an then project*/
        ProjectTableScanRule.INSTANCE,
        ProjectMergeRule.INSTANCE,
        //aggregate rules
        AggregateRemoveRule.INSTANCE,
        AggregateJoinTransposeRule.EXTENDED,
        AggregateProjectMergeRule.INSTANCE,
        AggregateProjectPullUpConstantsRule.INSTANCE,
        AggregateExpandDistinctAggregatesRule.INSTANCE,
        AggregateReduceFunctionsRule.INSTANCE,
//        //join rules
//        JoinToMultiJoinRule.INSTANCE,
//        LoptOptimizeJoinRule.INSTANCE,
//        MultiJoinOptimizeBushyRule.INSTANCE,
//        JoinPushThroughJoinRule.RIGHT,
//        JoinPushThroughJoinRule.LEFT,
//        /*choose between right and left*/
//        JoinPushExpressionsRule.INSTANCE,
//        JoinAssociateRule.INSTANCE,
//        JoinCommuteRule.INSTANCE,
        // simplify expressions rules
        ReduceExpressionsRule.CALC_INSTANCE,
        ReduceExpressionsRule.FILTER_INSTANCE,
        ReduceExpressionsRule.PROJECT_INSTANCE,
//        // prune empty results rules
//        PruneEmptyRules.FILTER_INSTANCE,
//        PruneEmptyRules.PROJECT_INSTANCE,
//        PruneEmptyRules.AGGREGATE_INSTANCE,
//        PruneEmptyRules.JOIN_LEFT_INSTANCE,
//        PruneEmptyRules.JOIN_RIGHT_INSTANCE,
        /* Sort Rules*/
        SortJoinTransposeRule.INSTANCE,
        SortProjectTransposeRule.INSTANCE,
        //SortRemoveRule.INSTANCE, //Too aggressive when triggered over enumerables; always removes Sort
        SortUnionTransposeRule.INSTANCE,
        /*Enumerable Rules*/
        EnumerableRules.ENUMERABLE_FILTER_RULE,
        EnumerableRules.ENUMERABLE_TABLE_SCAN_RULE,
        EnumerableRules.ENUMERABLE_PROJECT_RULE,
        EnumerableRules.ENUMERABLE_AGGREGATE_RULE,
        EnumerableRules.ENUMERABLE_JOIN_RULE,
        EnumerableRules.ENUMERABLE_MERGE_JOIN_RULE,
        EnumerableRules.ENUMERABLE_SEMI_JOIN_RULE,
        EnumerableRules.ENUMERABLE_SORT_RULE,
        EnumerableRules.ENUMERABLE_UNION_RULE,
        EnumerableRules.ENUMERABLE_INTERSECT_RULE,
        EnumerableRules.ENUMERABLE_MINUS_RULE
      )



      Frameworks.newConfigBuilder()
        .parserConfig(SqlParser.configBuilder().setLex(Lex.MYSQL).build())
        .defaultSchema(schema)
        .operatorTable(SqlStdOperatorTable.instance())
        //.traitDefs(traitDefs)
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
      try {
        val sqlNode = planner.parse(query)
        val validatedSqlNode = planner.validate(sqlNode)
        planner.rel(validatedSqlNode).project
      }
      catch {
        case e: SqlParseException =>
          throw new RuntimeException("Query parsing error.", e)
      }
    }

}
