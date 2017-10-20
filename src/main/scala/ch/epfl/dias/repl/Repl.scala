package ch.epfl.dias.repl

import java.io.PrintWriter
import java.sql.{Connection, DriverManager, Statement}
import java.util.Properties

import ch.epfl.dias.emitter.{PlanConversionException, PlanToJSON}
import ch.epfl.dias.sql.QueryToPlan
import com.google.common.io.Resources
import org.apache.calcite.adapter.enumerable.EnumerableConvention
import org.apache.calcite.jdbc.CalciteConnection
import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.plan.hep.{HepPlanner, HepProgramBuilder}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.sql.SqlExplainLevel
import org.json4s.JsonAST.JValue

import scala.io.{BufferedSource, Source, StdIn}

object Repl extends App {

  while (true) {
    print("sql > ")
    val input = StdIn.readLine()

      if (input == "exit" || input == "quit") {
        System.exit(0)
      }
      //Setup Connection
      Class.forName("org.apache.calcite.jdbc.Driver")
      val info = new Properties
      info.setProperty("lex", "JAVA")
      //Getting the actual content of schema.json
      //String schemaRaw = Resources.toString(QueryToPlan.class.getResource("/schema.json"), Charset.defaultCharset());

      //val schemaPath = /*getClass.getResource("schema.json").getPath*/ Resources.getResource("schema.json").getPath
      //TODO Not the cleanest way to provide this path, but sbt crashes otherwise
      val schemaPath: String = new java.io.File(".").getCanonicalPath+"/src/main/resources/schema.json"
      val connection = DriverManager.getConnection("jdbc:calcite:model=" + schemaPath, info)
      val calciteConnection = connection.unwrap(classOf[CalciteConnection])
      val rootSchema = calciteConnection.getRootSchema.getSubSchema("SALES") //or SSB
      val statement = connection.createStatement

      //Parse, validate, optimize query
      val queryPlanner: QueryToPlan = new QueryToPlan(rootSchema)
      val rel = queryPlanner.getLogicalPlan(input)
      val traitSet = queryPlanner.planner.getEmptyTraitSet.replace(EnumerableConvention.INSTANCE)
      var logicalPlan: RelNode = queryPlanner.planner.transform(0, traitSet, rel)
      //Heuristics optimizer
      val hepProgramBuilder = new HepProgramBuilder
      val hepPlanner = new HepPlanner(hepProgramBuilder.build)
      //Applying rules
      hepPlanner.setRoot(logicalPlan)
      logicalPlan = hepPlanner.findBestExp

      System.out.println("Calcite Plan:")
      System.out.println(RelOptUtil.toString(logicalPlan, SqlExplainLevel.EXPPLAN_ATTRIBUTES))

      //Emitting JSON equivalent of produced plan
      try {
        System.out.println("JSON Serialization:")
        val planJSON: JValue = PlanToJSON.emit(logicalPlan)
        val planStr: String = PlanToJSON.jsonToString(planJSON)
        System.out.println(planStr)
        val fileName = "current.json"
        val out = new PrintWriter(fileName, "UTF-8")
        //print to file
        try {
          out.print(planStr)
        } finally {
          out.close()
        }
      } catch {
        case c: PlanConversionException => {
          c.printStackTrace()
          System.exit(-1)
        }
      }
  }
}

/* Query Examples */
//val query = "select MAX(sales.PRICE) " +
//"from sales " +
//"JOIN item ON (item.sales_id = sales.id) " +
//"where name <> 'Sally' AND item.sales_id > 8 AND item.sales_id > 9 " +
//"GROUP BY item.id + 114, item.sales_id"

//val query_ = "select MAX(sales.PRICE), MIN(sales.PRICE) + 9 " +
//"from sales " +
//"JOIN item ON (item.sales_id = sales.id) " +
//"where name <> 'Sally' AND item.sales_id > 8 AND item.sales_id > 9 " +
//"GROUP BY sales.id + 114, sales.id + 119"