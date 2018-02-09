package ch.epfl.dias.repl

import java.io.PrintWriter
import java.net.URL
import java.sql.{Connection, DriverManager, Statement}
import java.util.Properties

import ch.epfl.dias.emitter.{PlanConversionException, PlanToJSON}
import ch.epfl.dias.sql.QueryToPlan
import com.google.common.io.Resources
import org.apache.calcite.adapter.enumerable.EnumerableConvention
import org.apache.calcite.interpreter.InterpretableConvention
import org.apache.calcite.jdbc.CalciteConnection
import org.apache.calcite.plan.{RelOptUtil, RelTraitSet}
import org.apache.calcite.plan.hep.{HepPlanner, HepProgramBuilder}
import org.apache.calcite.rel.RelNode
import org.apache.calcite.sql.SqlExplainLevel
import org.json4s.JsonAST.JValue
import org.apache.calcite.sql.parser.SqlParseException
import org.apache.calcite.sql.validate.SqlValidatorException
import org.apache.calcite.tools.ValidationException

import scala.io.{BufferedSource, Source, StdIn}

object Repl extends App {

  //Setup Connection
  Class.forName("org.apache.calcite.jdbc.Driver")
  val info = new Properties
  info.setProperty("lex", "JAVA")
  //Getting the actual content of schema.json
  //String schemaRaw = Resources.toString(QueryToPlan.class.getResource("/schema.json"), Charset.defaultCharset());

  /*
  //Getting the actual model doesn't do us any good, unless we put it together programmatically on our own
  //See https://calcite.apache.org/docs/model.html
  var schemaPath : String = getClass.getResource("/schema.json").getPath
  import java.io.InputStream
  import org.apache.commons.io.IOUtils
  val is = getClass.getResourceAsStream("/schema.json")
  val model = IOUtils.toString(is)
  */

  //TODO Not the cleanest way to provide this path, but sbt crashes otherwise. Incompatible with assembly jar
  val schemaPath: String = new java.io.File(".").getCanonicalPath+"/src/main/resources/schema.json"
  val connection = DriverManager.getConnection("jdbc:calcite:model=" + schemaPath, info)
  val calciteConnection: CalciteConnection = connection.unwrap(classOf[CalciteConnection])
  val rootSchema = calciteConnection.getRootSchema.getSubSchema("SSB") //or SALES
  val statement = connection.createStatement

  while (true) {
    print("sql > ")
    val input = StdIn.readLine()

    if (input == null || input == "" || input == "exit" || input == "quit") {
      System.exit(0)
    }

    try {
      //Parse, validate, optimize query
      val queryPlanner: QueryToPlan = new QueryToPlan(rootSchema)
      val rel: RelNode = queryPlanner.getLogicalPlan(input)

      System.out.println("Calcite Logical Plan:")
      System.out.println(RelOptUtil.toString(rel, SqlExplainLevel.EXPPLAN_ATTRIBUTES))

      val traitSet: RelTraitSet = queryPlanner.planner.getEmptyTraitSet.replace(EnumerableConvention.INSTANCE)
//      val traitSet = queryPlanner.planner.getEmptyTraitSet.replace(InterpretableConvention.INSTANCE)
      var logicalPlan: RelNode = queryPlanner.planner.transform(0, traitSet, rel)
      //Heuristics optimizer
      /*val hepProgramBuilder = new HepProgramBuilder
      val hepPlanner = new HepPlanner(hepProgramBuilder.build)
      //Applying rules
      hepPlanner.setRoot(logicalPlan)
      logicalPlan = hepPlanner.findBestExp*/

      System.out.println("Calcite Physical Plan:")
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
    } catch {
      case e: SqlParseException =>
        System.out.println("Query parsing error: " + e.getMessage)
      case e: ValidationException =>
        System.out.println("Query parsing error: " + e.getMessage)
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