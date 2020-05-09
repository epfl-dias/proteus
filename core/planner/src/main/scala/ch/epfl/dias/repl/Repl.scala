package ch.epfl.dias.repl

import java.sql.DriverManager
import java.util.Properties

import org.apache.calcite.avatica.jdbc.JdbcMeta
import org.apache.calcite.avatica.remote.Driver.Serialization
import org.apache.calcite.avatica.remote.LocalService
import org.apache.calcite.avatica.server.HttpServer

import scala.sys.process.Process

object Repl extends App {
  Class.forName("ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver")

  val arglist = args.toList
  val defaultMock = new java.io.File(".").getCanonicalPath+"/src/main/resources/mock.csv"
  val defaultSchema = new java.io.File(".").getCanonicalPath+"/src/main/resources/schema.json"

  type OptionMap = Map[Symbol, Any]

  def nextOption(map: OptionMap, list: List[String]): OptionMap = {
    def isSwitch(s: String) = (s(0) == '-')

    list match {
      case Nil => map
      case "--server" :: tail =>
        nextOption(map ++ Map('server -> true), tail)
      case "--echo-results" :: tail =>
        nextOption(map ++ Map('echoResults -> true), tail)
      case "--port" :: value :: tail =>
        nextOption(map ++ Map('port -> value.toInt), tail)
      case "--cpuonly" :: value :: tail =>
        nextOption(map ++ Map('cpuonly -> true), tail)
      case "--onlyinit" :: value :: tail =>
        nextOption(map ++ Map('onlyinit -> true), tail)
      case "--timings-csv" :: value :: tail =>
        nextOption(map ++ Map('timingscsv -> true), tail)
      case "--mockfile" :: value :: tail =>
        nextOption(map ++ Map('mockfile -> value) ++ Map('mock -> true), tail)
      case "--planfile" :: value :: tail =>
        nextOption(map ++ Map('planfile -> value), tail)
      case "--gpudop" :: value :: tail =>
        nextOption(map ++ Map('gpudop -> value.toInt), tail)
      case "--cpudop" :: value :: tail =>
        nextOption(map ++ Map('cpudop -> value.toInt), tail)
      case "--mock" :: tail =>
        nextOption(map ++ Map('mock -> true), tail)
      case string :: Nil => nextOption(map ++ Map('schema -> string), list.tail)
      case option :: tail => println("Unknown option " + option)
        println("Usage: [--server [--port]] [--echo-results] [--timings-csv] [--planfile <path-to-write-plan>] [--mockfile <path-to-mock-file>|--mock] [path-to-schema.json]")
        System.exit(1)
        null
    }
  }

  var executor_server = "./proteusmain-server"

  val topology = Process(executor_server + " --query-topology").!!
  val gpudop_regex = """gpu  count: (\d+)""".r.unanchored
  val detected_gpudop = topology match {
    case gpudop_regex(g) => g.toInt
    case _ => 0
  }
  val cpudop_regex = """core count: (\d+)""".r.unanchored
  val detected_cpudop = topology match {
    case cpudop_regex(c) => c.toInt/2             // We want by default to ignore hyperthreads
    case _ => 48
  }

  System.out.println(topology)

  val options = nextOption(Map(
                                'server -> false, 'port -> 8081, 'cpudop -> detected_cpudop, 'gpudop -> detected_gpudop,
                                'echoResults -> false, 'mock -> false, 'timings -> true, 'timingscsv -> false,
                                'mockfile -> defaultMock,
                                'cpuonly -> (detected_gpudop <= 0), 'planfile -> "plan.json", 'schema -> defaultSchema,
                                'onlyinit -> false, 'printplan -> false,
                              ), arglist)

  System.out.println(options);

  var mockfile    = options('mockfile   ).asInstanceOf[String ]
  var isMockRun   = options('mock       ).asInstanceOf[Boolean]
  var echoResults = options('echoResults).asInstanceOf[Boolean]
  var planfile    = options('planfile   ).asInstanceOf[String ]
  var printplan   = options('printplan  ).asInstanceOf[Boolean]
  var timingscsv  = options('timingscsv ).asInstanceOf[Boolean]
  var timings     = options('timings    ).asInstanceOf[Boolean]


  var cpus_on     = true //options('cpuonly    ).asInstanceOf[Boolean]
  var gpus_on     = true
  var cpudop      = options('cpudop     ).asInstanceOf[Int    ]
  var gpudop      = options('gpudop     ).asInstanceOf[Int    ]

  //default to hybrid execution
  setHybrid()
  //check if the user requested cpuonly execution (or no GPUs are available)
  if (options('cpuonly).asInstanceOf[Boolean]) setCpuonly()

  def isHybrid() = cpus_on && gpus_on
  def setHybrid(): Unit = {
    cpus_on = true
    gpus_on = true
  }

  def isCpuonly() = cpus_on && !gpus_on
  def setCpuonly(): Unit = {
    cpus_on = true
    gpus_on = false
  }

  def isGpuonly() = !cpus_on && gpus_on
  def setGpuonly(): Unit = {
    cpus_on = false
    gpus_on = true
  }

  /*
  //Getting the actual model doesn't do us any good, unless we put it together progtln("Unknown option " + option)         println("Usage: [--server [--port]] [--echo-results] [--planfile <path-to-write-plan>] [--mockfile <path-to-mock-file>|--mock] [path-to-schema.json]")         System.exit(1)rammatically on our own
  //See https://calcite.apache.org/docs/model.html
  var schemaPath : String = getClass.getResource("/schema.json").getPath
  import java.io.InputStream
  import org.apache.commons.io.IOUtils
  val is = getClass.getResourceAsStream("/schema.json")
  val model = IOUtils.toString(is)
  */
  //TODO Not the cleanest way to provide this path, but sbt crashes otherwise. Incompatible with assembly jar
  val schemaPath: String = options.get('schema).get.asInstanceOf[String]

  if (options.get('onlyinit).get.asInstanceOf[Boolean]) {
    // Done, return from object initialization
  } else if (options.get('server).get.asInstanceOf[Boolean]) {
    val meta = new JdbcMeta("jdbc:pelago:model=" + schemaPath)
    val service = new LocalService(meta)
    val server = new HttpServer.Builder().withHandler(service, Serialization.PROTOBUF).withPort(options.get('port).get.asInstanceOf[Int]).build();

    server.start();
    server.join();
  } else {
    //Setup Connection
    //  Class.forName("org.apache.calcite.jdbc.Driver")
    Class.forName("ch.epfl.dias.calcite.adapter.pelago.jdbc.Driver")
    val info = new Properties
    //  info.setProperty("lex", "JAVA")
    //Getting the actual content of schema.json
    //String schemaRaw = Resources.toString(QueryToPlan.class.getResource("/schema.json"), Charset.defaultCharset());
    var input : String = null;
    val connection = DriverManager.getConnection("jdbc:pelago:model=" + schemaPath, info)
    //  val calciteConnection: CalciteConnection = connection.unwrap(classOf[CalciteConnection])
    //  val rootSchema = calciteConnection.getRootSchema.getSubSchema("SSB") //or SALES
    //  val statement = connection.createStatement
    connection.setSchema("SSB");

    //  while (true) {
    ////  {
    //    print("sql > ")
    //    val input = StdIn.readLine()
    //
    //    if (input == null || input == "" || input == "exit" || input == "quit") {
    //      System.exit(0)
    //    }


//          input = "select count(*) " + //sum(lo_revenue - lo_supplycost) as profit " +
//            "from ssbm_lineorder, ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part " +
//            "where lo_custkey = c_custkey " +
//            "and lo_suppkey = s_suppkey " +
//            "and lo_partkey = p_partkey " +
//            "and lo_orderdate = d_datekey " +
//            "and c_region = 'AMERICA' " +
//            "and s_region = 'AMERICA' " +
//            "and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') " //+
////            "group by d_year";//+
//    //        "order by d_year, c_nation";

    input =
      "select sum(lo_revenue), d_year, p_brand1 " +
      "from ssbm_date, ssbm_lineorder, ssbm_part, ssbm_supplier " +
      "where lo_orderdate = d_datekey " +
      " and lo_partkey = p_partkey " +
      " and lo_suppkey = s_suppkey " +
      " and p_category = 'MFGR#12' " +
      " and s_region = 'AMERICA' " +
      "group by d_year, p_brand1 " +
      "order by d_year, p_brand1"

//    input = "select max(d_yearmonthnum), d_year from ssbm_date group by d_year order by d_year";
//    input = "select * from employees";

//    input = "create table Test1234(a integer, b integer) jplugin `{\"plugin\":{ \"type\":\"block\", \"linehint\":200000 }, \"file\":\"/inputs/csv.csv\"}`";
//    connection.createStatement().execute(input);
//
//    val md = connection.getMetaData
//    val rs = md.getTables(null, null, "%", null)
//    while (rs.next)
//      println(rs.getString(3))
//
//    input = "select count(*) from Test1234"

    var resultSet = connection.createStatement().executeQuery("explain plan for " + input)
    //    connection.createStatement().execute(input);
    //    var resultSet = connection.getMetaData.getTables(null, null, "%", null)
    var rsmd = resultSet.getMetaData
    var columnsNumber = rsmd.getColumnCount
    while ( {
      resultSet.next
    }) {
      var i = 1
      while ( {
        i <= columnsNumber
      }) {
        if (i > 1) System.out.print(",  ")
        val columnValue = resultSet.getString(i)
        System.out.print(columnValue + " " + rsmd.getColumnName(i))

        {
          i += 1;
          i - 1
        }
      }
      System.out.println("")
    }

    System.out.println("")
    System.out.println("")
    System.out.println("")
    System.out.println("")
    System.out.println("")
    System.out.println("rr")
    System.out.println("rr")
    System.out.println("rr")
    resultSet = connection.createStatement().executeQuery(input)
    rsmd = resultSet.getMetaData
    columnsNumber = rsmd.getColumnCount
    while ( {
      resultSet.next
    }) {
      var i = 1
      while ( {
        i <= columnsNumber
      }) {
        if (i > 1) System.out.print(",  ")
        val columnValue = resultSet.getString(i)
        System.out.print(columnValue + " " + rsmd.getColumnName(i))

        {
          i += 1;
          i - 1
        }
      }
      System.out.println("")
    }
    //    connection.createStatement().executeQuery(input)
    System.exit(0);
    //    val input = "select d_year, c_nation, sum(lo_revenue - lo_supplycost) as profit " +
    //      "from ssbm_date, ssbm_customer, ssbm_supplier, ssbm_part, ssbm_lineorder " +
    //      "where lo_custkey = c_custkey " +
    //      "and lo_suppkey = s_suppkey "+
    //      "and lo_partkey = p_partkey "+
    //      "and lo_orderdate = d_datekey "+
    //      "and c_region = 'AMERICA' "+
    //      "and s_region = 'AMERICA' "+
    //      "and (p_mfgr = 'MFGR#1' or p_mfgr = 'MFGR#2') " +
    //      "group by d_year, c_nation "+
    //      "order by d_year, c_nation";


    //  val input = "select sum(lo_revenue - lo_supplycost) as profit " +
    //    "from ssbm_customer, ssbm_supplier, ssbm_lineorder " +
    //    "where lo_custkey = c_custkey " +
    //    "and lo_suppkey = s_suppkey ";
    //    val input = "select sum(lo_revenue) as lo_revenue, d_year, p_brand1 from ssbm_lineorder, ssbm_date, ssbm_part, ssbm_supplier where lo_orderdate = d_datekey  and lo_partkey = p_partkey  and lo_suppkey = s_suppkey  and p_brand1 = 'MFGR#2239'  and s_region = 'EUROPE'  group by d_year, p_brand1 order by d_year asc, p_brand1 desc"
    //
    //    try {
    //      //Parse, validate, optimize query
    //      val queryPlanner: QueryToPlan = new QueryToPlan(rootSchema)
    //      val rel: RelNode = queryPlanner.getLogicalPlan(input)
    ////      val rel2 = .copy(queryPlanner.planner.getEmptyTraitSet.plus(RelDistributions.SINGLETON), )
    ////      val rel2: RelNode = RelDistributionTraitDef.INSTANCE.convert(queryPlanner.planner, rel, RelDistributions.SINGLETON, true) //RelNode = queryPlanner.planner.transform(0, queryPlanner.planner.getEmptyTraitSet.plus(RelDistributions.RANDOM_DISTRIBUTED), rel)
    //
    //      System.out.println("Calcite Logical Plan:")
    //      System.out.println(RelOptUtil.toString(rel, SqlExplainLevel.EXPPLAN_ATTRIBUTES))
    ////      EnumerableConvention.INSTANCE
    ////      System.exit(0)
    ////      val traitSet: RelTraitSet = queryPlanner.planner.getEmptyTraitSet.replace(RelDistributions.RANDOM_DISTRIBUTED)
    //      val traitSet: RelTraitSet = queryPlanner.planner.getEmptyTraitSet.replace(EnumerableConvention.INSTANCE) //.plus(RelDistributions.SINGLETON) //FIXME: do we need a sequential output ?
    //      //      val traitSet = queryPlanner.planner.getEmptyTraitSet.replace(InterpretableConvention.INSTANCE)
    //      var logicalPlan: RelNode = queryPlanner.planner.transform(0, traitSet, rel)
    ////      var logicalPlan: RelNode = RelDistributionTraitDef.INSTANCE.convert(queryPlanner.planner., rel, RelDistributions.SINGLETON, true)//.transform(0, traitSet, rel)
    //      //Heuristics optimizer
    //      /*val hepProgramBuilder = new HepProgramBuilder
    //      val hepPlanner = new HepPlanner(hepProgramBuilder.build)
    //      //Applying rules
    //      hepPlanner.setRoot(logicalPlan)
    //      logicalPlan = hepPlanner.findBestExp*/
    //
    //      System.out.println("Calcite Physical Plan:")
    //      System.out.println(RelOptUtil.toString(logicalPlan, SqlExplainLevel.EXPPLAN_ATTRIBUTES))
    //      //Emitting JSON equivalent of produced plan
    //
    ////      try {
    ////        System.out.println("JSON Serialization:")
    ////        val planJSON: JValue = PlanToJSON.emit(logicalPlan)
    ////        val planStr: String = PlanToJSON.jsonToString(planJSON)
    ////        System.out.println(planStr)
    ////        val fileName = "current.json"
    ////        val out = new PrintWriter(fileName, "UTF-8")
    ////        //print to file
    ////        try {
    ////          out.print(planStr)
    ////        } finally {
    ////          out.close()
    ////        }
    ////      } catch {
    ////        case c: PlanConversionException => {
    ////          c.printStackTrace()
    ////          System.exit(-1)
    ////        }
    ////      }
    ////    } catch {
    ////      case e: SqlParseException =>
    ////        System.out.println("Query parsing error: " + e.getMessage)
    ////      case e: ValidationException =>
    ////        System.out.println("Query parsing error: " + e.getMessage)
    ////    }
    //  }
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