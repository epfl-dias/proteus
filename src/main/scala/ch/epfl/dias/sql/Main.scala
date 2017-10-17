package ch.epfl.dias.sql

import java.sql.{Connection, DriverManager, Statement}
import java.util.Properties

import com.google.common.io.Resources
import org.apache.calcite.adapter.enumerable.EnumerableConvention
import org.apache.calcite.jdbc.CalciteConnection
import org.apache.calcite.plan.{RelOptUtil, RelTraitSet}
import org.apache.calcite.plan.hep.{HepPlanner, HepProgramBuilder}
import org.apache.calcite.sql.SqlExplainLevel

object Main extends App {
  Class.forName("org.apache.calcite.jdbc.Driver")
  val info = new Properties
  info.setProperty("lex", "JAVA")
  //Getting the actual content of schema.json
  //String schemaRaw = Resources.toString(QueryToPlan.class.getResource("/schema.json"), Charset.defaultCharset());
  val schemaPath = Resources.getResource("schema.json").getPath
  val connection = DriverManager.getConnection("jdbc:calcite:model=" + schemaPath, info)
  val calciteConnection = connection.unwrap(classOf[CalciteConnection])

  val rootSchema = calciteConnection.getRootSchema.getSubSchema(connection.getSchema)

  val statement = connection.createStatement

  val queryPlanner : QueryToPlan = new QueryToPlan(rootSchema)

  val query = "select sales.ID from sales JOIN item ON (sales.id = item.sales_id) " + "where name <> 'Sally' AND item.sales_id > 8 AND item.sales_id > 9 "
  val rel = queryPlanner.getLogicalPlan(query)

  val traitSet = queryPlanner.planner.getEmptyTraitSet.replace(EnumerableConvention.INSTANCE)
  var logicalPlan = queryPlanner.planner.transform(0, traitSet, rel)

  /* HEURISTICS OPTIMIZER */
  val hepProgramBuilder = new HepProgramBuilder
  val hepPlanner = new HepPlanner(hepProgramBuilder.build)

  System.out.println("Applying rules...")
  hepPlanner.setRoot(logicalPlan)
  logicalPlan = hepPlanner.findBestExp

  hepPlanner.setRoot(logicalPlan)
  logicalPlan = hepPlanner.findBestExp

  System.out.println(RelOptUtil.toString(logicalPlan, SqlExplainLevel.EXPPLAN_ATTRIBUTES))
}
